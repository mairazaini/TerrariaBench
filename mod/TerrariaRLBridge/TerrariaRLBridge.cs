using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Terraria;
using Terraria.Audio;
using Terraria.GameContent;
using Terraria.ID;
using Terraria.ModLoader;
using Terraria.UI;

namespace TerrariaRLBridge
{
    // HUMAN-PARITY BRIDGE (protocol 2)
    //
    // This mod is a keyboard, a mouse and a monitor for an external agent. Nothing more.
    //   * Sight:    the agent receives the frame the game actually rendered (lighting, darkness and UI included).
    //   * Hands:    movement keys, mouse buttons, a cursor position, hotbar keys, inventory/crafting/chest UI actions.
    //   * No cheats: the mod never edits tiles, spawns items or NPCs, changes time, heals, pulls items or aims for
    //               the agent. Every world change goes through the same game systems a human player uses.
    // Anything the game HUD does not show a human (coordinates, tile grid, enemy list, boss HP) is sent only inside the
    // "referee" block, which the Python side uses for scoring and must never forward to a model.
    public class TerrariaRLBridgeMod : Mod
    {
    }

    internal sealed class AgentControlSnapshot
    {
        public bool Left;
        public bool Right;
        public bool Up;
        public bool Down;
        public bool Jump;
        public bool UseItem;
        public bool UseTile;
        public bool Hook;
        public bool QuickHeal;
        public bool SmartCursor;
        public int Hotbar;            // 1-10 selects that hotbar key; 0 keeps the current selection
        public float CursorDx;        // cursor offset from the player's centre, in tiles (right is positive)
        public float CursorDy;        // cursor offset from the player's centre, in tiles (down is positive)
        public int HoldTicks = 20;    // how long the keys stay pressed before they are released
        public string InventoryCmd = "";   // "open" | "close" | ""
        public string Craft = "";          // result item name; needs the inventory open and a currently available recipe
        public string InventoryItem = "";  // inventory item to equip (armor/accessory) or move to the selected hotbar slot
        public bool LootAll;               // press "Loot All" on the chest that is currently open
        public bool AimAssist;             // tool chosen by the decision model: cursor follows nearest visible enemy
        public string SupervisorLine = "";
        public string VisionLine = "";
        public string DecisionLine = "";
        public int ExpireTick;
        public bool OneShotsDone;
        public DateTime ReceivedAtUtc = DateTime.MinValue;
    }

    /// <summary>
    /// Presses exactly the keys the decision model asked for. No item selection, no automatic attacks. The one tool is
    /// aim assist: only when the decision model switches it on, the cursor follows the nearest enemy that is on
    /// screen and lit (something the player could see). It never clicks by itself.
    /// </summary>
    public class RLAgentPlayer : ModPlayer
    {
        private bool _wasDead;
        private bool _wasUsingItem;
        private bool _wasJumping;
        public int DeathCount { get; private set; }

        public override void SetControls()
        {
            if (Main.dedServ || Main.gameMenu || Player.whoAmI != Main.myPlayer)
                return;

            var snap = MeteredBridgeSystem.LatestControls;
            // No agent connected (or it went silent): leave the controls alone so a human can play normally.
            if (snap == null || (DateTime.UtcNow - snap.ReceivedAtUtc).TotalSeconds > 30.0)
                return;

            // Keys are released once the requested hold time is over; the agent must decide again to keep acting.
            bool held = MeteredBridgeSystem.TickCounter <= snap.ExpireTick;

            Player.controlLeft = held && snap.Left;
            Player.controlRight = held && snap.Right;
            Player.controlUp = held && snap.Up;
            Player.controlDown = held && snap.Down;
            Player.controlUseTile = held && snap.UseTile;
            Player.controlHook = held && snap.Hook;
            Player.controlQuickHeal = held && snap.QuickHeal;

            if (held && snap.Jump)
            {
                // A held jump key does not re-trigger on landing; re-press it like a player tapping Space.
                if (_wasJumping && Player.velocity.Y == 0f)
                {
                    Player.controlJump = false;
                    Player.releaseJump = true;
                    _wasJumping = false;
                }
                else
                {
                    Player.controlJump = true;
                    _wasJumping = true;
                }
            }
            else
            {
                Player.controlJump = false;
                _wasJumping = false;
            }

            if (held && snap.UseItem)
            {
                // Re-click between swings so non-autoReuse items repeat, like a player clicking repeatedly.
                if (_wasUsingItem && Player.itemAnimation == 0)
                {
                    Player.controlUseItem = false;
                    Player.releaseUseItem = true;
                    _wasUsingItem = false;
                }
                else
                {
                    Player.controlUseItem = true;
                    if (Player.itemAnimation > 0)
                        _wasUsingItem = true;
                }
            }
            else
            {
                Player.controlUseItem = false;
                _wasUsingItem = false;
            }

            // Hotbar number key. Like the real game, the selection cannot change mid-swing.
            if (snap.Hotbar >= 1 && snap.Hotbar <= 10 && Player.itemAnimation == 0 && Player.itemTime == 0)
                Player.selectedItem = snap.Hotbar - 1;

            // Smart Cursor is a normal game feature (Ctrl); it is on only when the model turns it on.
            Main.SmartCursorWanted_Mouse = snap.SmartCursor;

            // Mouse cursor: a plain offset from the player chosen by the model. The game's own reach limits apply.
            Vector2 aimWorld = Player.Center + new Vector2(snap.CursorDx * 16f, snap.CursorDy * 16f);
            if (held && snap.AimAssist)
            {
                NPC target = MeteredBridgeSystem.NearestVisibleEnemy(Player);
                if (target != null)
                    aimWorld = target.Center;
            }
            Main.mouseX = (int)Math.Clamp(aimWorld.X - Main.screenPosition.X, 4, Main.screenWidth - 4);
            Main.mouseY = (int)Math.Clamp(aimWorld.Y - Main.screenPosition.Y, 4, Main.screenHeight - 4);
        }

        public override void PostUpdate()
        {
            if (Main.dedServ || Player.whoAmI != Main.myPlayer)
                return;

            if (Player.dead && !_wasDead)
                DeathCount++;
            _wasDead = Player.dead;
        }
    }

    public class MeteredBridgeSystem : ModSystem
    {
        // Three model tiers answer each tick, so allow far more than a frame's worth of latency.
        private static readonly HttpClient Http = new HttpClient { Timeout = TimeSpan.FromSeconds(45) };
        private const string BridgeUrl = "http://127.0.0.1:8765/real_game_tick";
        private const int MinPollTicks = 8;
        private const int ShotWidth = 640;
        private const int MaxHoldTicks = 180;

        internal static volatile AgentControlSnapshot LatestControls;
        internal static int TickCounter;

        private volatile bool _inFlight;
        private bool _wantCapture;
        private int _wantCaptureSince;
        private int _lastPostTick = -MinPollTicks;
        private volatile string _bridgeError = "";
        private DateTime _lastReplyUtc = DateTime.MinValue;
        private int _replies;
        private readonly List<string> _events = new List<string>();

        private Color[] _backBuffer;
        private Color[] _shotPixels;
        private Texture2D _shotTexture;

        public override void Load()
        {
            if (!Main.dedServ)
                Main.OnPostDraw += OnPostDraw;
        }

        public override void Unload()
        {
            if (!Main.dedServ)
                Main.OnPostDraw -= OnPostDraw;
            LatestControls = null;
            _shotTexture = null;
        }

        public override void OnWorldLoad()
        {
            TickCounter = 0;
            _lastPostTick = -MinPollTicks;
            _wantCapture = false;
            _replies = 0;
            _bridgeError = "";
            _lastReplyUtc = DateTime.MinValue;
            _events.Clear();
            LatestControls = null;

            // Keep simulating and rendering when the window is not focused so unattended runs do not stall.
            if (!Main.dedServ && Main.instance != null)
                Main.instance.InactiveSleepTime = TimeSpan.Zero;
        }

        public override void PostUpdateEverything()
        {
            if (Main.dedServ || Main.gameMenu)
                return;

            Player p = Main.LocalPlayer;
            if (p == null || !p.active)
                return;

            TickCounter++;

            var snap = LatestControls;
            if (snap != null && !snap.OneShotsDone)
            {
                snap.OneShotsDone = true;
                if (!p.dead)
                    ApplyUiActions(p, snap);
            }

            if (!_inFlight && !_wantCapture && TickCounter - _lastPostTick >= MinPollTicks)
            {
                _wantCapture = true;
                _wantCaptureSince = TickCounter;
            }

            // Nothing was drawn for a while (minimised window): report honestly that the agent has no picture.
            if (_wantCapture && !_inFlight && TickCounter - _wantCaptureSince > 120)
            {
                _wantCapture = false;
                PostTickAsync(p, null, "no frame rendered (window minimised or hidden)");
            }
        }

        // Runs on the main thread right after the game finished drawing the frame the player would see.
        private void OnPostDraw(GameTime gameTime)
        {
            if (!_wantCapture || _inFlight || Main.gameMenu)
                return;
            Player p = Main.LocalPlayer;
            if (p == null || !p.active)
                return;

            _wantCapture = false;
            string shot = null;
            string error = null;
            try
            {
                shot = CaptureScreenshot();
            }
            catch (Exception e)
            {
                error = $"screen capture failed: {e.GetType().Name}: {e.Message}";
            }
            PostTickAsync(p, shot, error);
        }

        /// <summary>
        /// Reads the rendered back buffer (what is on the monitor: lighting, darkness, HUD, open menus) and returns a
        /// downscaled PNG. This is the agent's only view of the world.
        /// </summary>
        private int _shotW;
        private int _shotH;
        private float _shotScale;

        private string CaptureScreenshot()
        {
            GraphicsDevice gd = Main.instance.GraphicsDevice;
            int w = gd.PresentationParameters.BackBufferWidth;
            int h = gd.PresentationParameters.BackBufferHeight;
            if (w <= 0 || h <= 0)
                throw new InvalidOperationException("empty back buffer");

            if (_backBuffer == null || _backBuffer.Length != w * h)
                _backBuffer = new Color[w * h];
            gd.GetBackBufferData(_backBuffer);

            int tw = Math.Min(ShotWidth, w);
            int th = Math.Max(1, (int)Math.Round(h * (tw / (double)w)));
            _shotW = tw;
            _shotH = th;
            _shotScale = tw / (float)w;
            if (_shotPixels == null || _shotPixels.Length != tw * th)
                _shotPixels = new Color[tw * th];

            for (int y = 0; y < th; y++)
            {
                int y0 = y * h / th;
                int y1 = Math.Min(h - 1, (y0 + ((y + 1) * h / th)) / 2);
                for (int x = 0; x < tw; x++)
                {
                    int x0 = x * w / tw;
                    int x1 = Math.Min(w - 1, (x0 + ((x + 1) * w / tw)) / 2);
                    Color a = _backBuffer[y0 * w + x0];
                    Color b = _backBuffer[y0 * w + x1];
                    Color c = _backBuffer[y1 * w + x0];
                    Color d = _backBuffer[y1 * w + x1];
                    _shotPixels[y * tw + x] = new Color(
                        (a.R + b.R + c.R + d.R) / 4,
                        (a.G + b.G + c.G + d.G) / 4,
                        (a.B + b.B + c.B + d.B) / 4,
                        255);
                }
            }

            if (_shotTexture == null || _shotTexture.IsDisposed || _shotTexture.Width != tw || _shotTexture.Height != th)
            {
                _shotTexture?.Dispose();
                _shotTexture = new Texture2D(gd, tw, th);
            }
            _shotTexture.SetData(_shotPixels);
            using var ms = new MemoryStream();
            _shotTexture.SaveAsPng(ms, tw, th);
            return Convert.ToBase64String(ms.ToArray());
        }

        // ------------------------------------------------------------------ UI actions (inventory, crafting, chests)

        private void ApplyUiActions(Player p, AgentControlSnapshot snap)
        {
            if (snap.InventoryCmd == "open" && !Main.playerInventory)
            {
                Main.playerInventory = true;
                SoundEngine.PlaySound(SoundID.MenuOpen);
                Recipe.FindRecipes();
            }
            else if (snap.InventoryCmd == "close" && Main.playerInventory)
            {
                Main.playerInventory = false;
                p.chest = -1;
                SoundEngine.PlaySound(SoundID.MenuClose);
            }

            if (!string.IsNullOrEmpty(snap.Craft))
                TryCraft(p, snap.Craft);
            if (!string.IsNullOrEmpty(snap.InventoryItem))
                TryUseInventoryItem(p, snap.InventoryItem);
            if (snap.LootAll)
            {
                if (p.chest != -1)
                {
                    ChestUI.LootAll();
                    AddEvent("Loot All pressed on the open chest");
                }
                else
                {
                    AddEvent("Loot All failed: no chest is open");
                }
            }
        }

        /// <summary>
        /// Crafts one batch through the game's own recipe list: the recipe must be available right now (materials in
        /// the inventory, required crafting station in range) and the inventory must be open on screen.
        /// </summary>
        private void TryCraft(Player p, string itemName)
        {
            if (!Main.playerInventory)
            {
                AddEvent($"craft '{itemName}' failed: inventory is closed");
                return;
            }

            Recipe.FindRecipes();
            for (int i = 0; i < Main.numAvailableRecipes; i++)
            {
                Recipe recipe = Main.recipe[Main.availableRecipe[i]];
                if (recipe?.createItem == null || recipe.createItem.IsAir)
                    continue;
                if (!string.Equals(recipe.createItem.Name, itemName, StringComparison.OrdinalIgnoreCase))
                    continue;

                Main.focusRecipe = i; // scroll the on-screen crafting list to what is being crafted
                Item crafted = recipe.createItem.Clone();
                crafted.Prefix(-1);
                int made = crafted.stack;
                string name = crafted.Name;
                recipe.Create(); // consumes the real ingredients
                PopupText.NewText(PopupTextContext.ItemCraft, crafted, made);
                SoundEngine.PlaySound(SoundID.Grab);

                Item leftover = p.GetItem(p.whoAmI, crafted, GetItemSettings.InventoryEntityToPlayerInventorySettings);
                if (leftover != null && !leftover.IsAir)
                    p.QuickSpawnItem(p.GetSource_Misc("TerrariaRLBridge_CraftOverflow"), leftover, leftover.stack);

                Recipe.FindRecipes();
                AddEvent($"crafted {name} x{made}");
                Main.NewText($"Agent crafted {name} x{made}", 134, 239, 172);
                return;
            }

            AddEvent($"craft '{itemName}' failed: recipe not available (missing materials or crafting station)");
        }

        /// <summary>
        /// Inventory click: armor and accessories are equipped, anything else is swapped into the selected hotbar slot.
        /// </summary>
        private void TryUseInventoryItem(Player p, string itemName)
        {
            if (!Main.playerInventory)
            {
                AddEvent($"inventory item '{itemName}' failed: inventory is closed");
                return;
            }

            int idx = -1;
            for (int pass = 0; pass < 2 && idx < 0; pass++)
            {
                int from = pass == 0 ? 10 : 0;
                int to = pass == 0 ? 50 : 10;
                for (int i = from; i < to; i++)
                {
                    Item it = p.inventory[i];
                    if (it != null && !it.IsAir && string.Equals(it.Name, itemName, StringComparison.OrdinalIgnoreCase))
                    {
                        idx = i;
                        break;
                    }
                }
            }
            if (idx < 0)
            {
                AddEvent($"inventory item '{itemName}' failed: not in inventory");
                return;
            }

            Item item = p.inventory[idx];
            string name = item.Name;
            int armorSlot = item.headSlot >= 0 ? 0 : (item.bodySlot >= 0 ? 1 : (item.legSlot >= 0 ? 2 : -1));
            if (armorSlot >= 0)
            {
                Utils.Swap(ref p.inventory[idx], ref p.armor[armorSlot]);
                AddEvent($"equipped {name}");
            }
            else if (item.accessory)
            {
                int free = -1;
                for (int s = 3; s < 8; s++)
                {
                    if (p.armor[s] == null || p.armor[s].IsAir)
                    {
                        free = s;
                        break;
                    }
                }
                if (free < 0)
                {
                    AddEvent($"equip {name} failed: no free accessory slot");
                    return;
                }
                Utils.Swap(ref p.inventory[idx], ref p.armor[free]);
                AddEvent($"equipped accessory {name}");
            }
            else
            {
                int sel = Math.Clamp(p.selectedItem, 0, 9);
                if (idx == sel)
                    return;
                Utils.Swap(ref p.inventory[idx], ref p.inventory[sel]);
                AddEvent($"moved {name} to hotbar slot {sel + 1}");
            }
            SoundEngine.PlaySound(SoundID.Grab);
            Recipe.FindRecipes();
        }

        private void AddEvent(string text)
        {
            _events.Add(text);
            if (_events.Count > 8)
                _events.RemoveAt(0);
        }

        // ------------------------------------------------------------------ bridge I/O

        private void PostTickAsync(Player p, string screenshotB64, string screenshotError)
        {
            string payload = SerializeState(p, screenshotB64, screenshotError);
            _inFlight = true;
            _lastPostTick = TickCounter;
            _ = Task.Run(async () =>
            {
                try
                {
                    using var content = new StringContent(payload, Encoding.UTF8, "application/json");
                    using var resp = await Http.PostAsync(BridgeUrl, content);
                    if (!resp.IsSuccessStatusCode)
                    {
                        _bridgeError = $"bridge HTTP {(int)resp.StatusCode}";
                        return;
                    }

                    string body = await resp.Content.ReadAsStringAsync();
                    using JsonDocument doc = JsonDocument.Parse(body);
                    JsonElement root = doc.RootElement;
                    if (!root.TryGetProperty("controls", out JsonElement ctrl))
                    {
                        _bridgeError = "bridge reply had no controls";
                        return;
                    }

                    int hold = Math.Clamp(GetInt(ctrl, "hold_ticks", 20), 1, MaxHoldTicks);
                    var snap = new AgentControlSnapshot
                    {
                        Left = GetBool(ctrl, "left"),
                        Right = GetBool(ctrl, "right"),
                        Up = GetBool(ctrl, "up"),
                        Down = GetBool(ctrl, "down"),
                        Jump = GetBool(ctrl, "jump"),
                        UseItem = GetBool(ctrl, "use_item"),
                        UseTile = GetBool(ctrl, "use_tile"),
                        Hook = GetBool(ctrl, "hook"),
                        QuickHeal = GetBool(ctrl, "quick_heal"),
                        SmartCursor = GetBool(ctrl, "smart_cursor"),
                        Hotbar = GetInt(ctrl, "hotbar", 0),
                        CursorDx = Math.Clamp(GetFloat(ctrl, "cursor_dx", 0f), -60f, 60f),
                        CursorDy = Math.Clamp(GetFloat(ctrl, "cursor_dy", 0f), -34f, 34f),
                        HoldTicks = hold,
                        InventoryCmd = GetStr(ctrl, "inventory", ""),
                        Craft = GetStr(ctrl, "craft", ""),
                        InventoryItem = GetStr(ctrl, "inventory_item", ""),
                        LootAll = GetBool(ctrl, "loot_all"),
                        AimAssist = GetBool(ctrl, "aim_assist"),
                        ExpireTick = TickCounter + hold,
                        ReceivedAtUtc = DateTime.UtcNow,
                    };
                    if (root.TryGetProperty("hud", out JsonElement hud) && hud.ValueKind == JsonValueKind.Object)
                    {
                        snap.SupervisorLine = GetStr(hud, "supervisor", "");
                        snap.VisionLine = GetStr(hud, "vision", "");
                        snap.DecisionLine = GetStr(hud, "decision", "");
                    }

                    LatestControls = snap;
                    _lastReplyUtc = DateTime.UtcNow;
                    _replies++;
                    _bridgeError = "";
                }
                catch (Exception e)
                {
                    // No bridge, no models, no actions: the agent simply stands still. Nothing plays on its behalf.
                    _bridgeError = $"bridge unreachable: {e.GetType().Name}";
                }
                finally
                {
                    _inFlight = false;
                }
            });
        }

        private static bool GetBool(JsonElement el, string name)
        {
            return el.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.True;
        }

        private static string GetStr(JsonElement el, string name, string fallback)
        {
            return el.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.String ? (v.GetString() ?? fallback) : fallback;
        }

        private static int GetInt(JsonElement el, string name, int fallback)
        {
            return el.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number && v.TryGetDouble(out double d) ? (int)Math.Round(d) : fallback;
        }

        private static float GetFloat(JsonElement el, string name, float fallback)
        {
            return el.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number && v.TryGetDouble(out double d) ? (float)d : fallback;
        }

        private string SerializeState(Player p, string screenshotB64, string screenshotError)
        {
            // ---- What a human can read off the game's own HUD ----
            var hotbar = new List<object>(10);
            for (int i = 0; i < 10; i++)
            {
                Item it = p.inventory[i];
                bool empty = it == null || it.IsAir;
                hotbar.Add(new { slot = i + 1, name = empty ? "" : it.Name, stack = empty ? 0 : it.stack });
            }

            bool invOpen = Main.playerInventory;
            Dictionary<string, int> inventory = null;
            List<string> equipment = null;
            List<object> craftable = null;
            if (invOpen)
            {
                // Only visible while the inventory screen is open, exactly like for a human player.
                inventory = new Dictionary<string, int>();
                for (int i = 10; i < 50; i++)
                {
                    Item it = p.inventory[i];
                    if (it == null || it.IsAir || string.IsNullOrEmpty(it.Name))
                        continue;
                    inventory[it.Name] = inventory.GetValueOrDefault(it.Name) + it.stack;
                }

                equipment = new List<string>();
                for (int i = 0; i < 8 && i < p.armor.Length; i++)
                {
                    Item it = p.armor[i];
                    if (it != null && !it.IsAir)
                        equipment.Add(it.Name);
                }

                craftable = new List<object>();
                var seen = new HashSet<string>();
                for (int i = 0; i < Main.numAvailableRecipes && craftable.Count < 60; i++)
                {
                    Recipe r = Main.recipe[Main.availableRecipe[i]];
                    if (r?.createItem == null || r.createItem.IsAir || !seen.Add(r.createItem.Name))
                        continue;
                    craftable.Add(new { name = r.createItem.Name, makes = r.createItem.stack });
                }
            }

            Dictionary<string, int> chest = null;
            if (p.chest >= 0 && p.chest < Main.maxChests && Main.chest[p.chest] != null)
            {
                chest = new Dictionary<string, int>();
                foreach (Item it in Main.chest[p.chest].item)
                {
                    if (it == null || it.IsAir || string.IsNullOrEmpty(it.Name))
                        continue;
                    chest[it.Name] = chest.GetValueOrDefault(it.Name) + it.stack;
                }
            }

            var buffs = new List<string>();
            for (int i = 0; i < p.buffType.Length; i++)
            {
                if (p.buffType[i] > 0 && p.buffTime[i] > 0)
                    buffs.Add(Lang.GetBuffName(p.buffType[i]));
            }

            var events = new List<string>(_events);
            _events.Clear();

            // Names only, and only of enemies the player could see right now. Positions stay in the picture.
            var visibleEnemies = new List<string>();
            for (int i = 0; i < Main.maxNPCs && visibleEnemies.Count < 8; i++)
            {
                if (IsVisibleEnemy(Main.npc[i]))
                    visibleEnemies.Add(Main.npc[i].GivenOrTypeName);
            }

            var snap = LatestControls;
            var hudBlock = new
            {
                hp = p.statLife,
                max_hp = p.statLifeMax2,
                mana = p.statMana,
                max_mana = p.statManaMax2,
                dead = p.dead,
                selected_slot = p.selectedItem + 1,
                held_item = p.HeldItem == null || p.HeldItem.IsAir ? "" : p.HeldItem.Name,
                hotbar,
                buffs,
                inventory_open = invOpen,
                inventory,
                equipment,
                craftable,
                chest_open = chest != null,
                chest,
                cursor = new { dx = snap?.CursorDx ?? 0f, dy = snap?.CursorDy ?? 0f },
                events,
                visible_enemies = visibleEnemies,
                // Image scale only (what a player judges by eye): lets the vision model convert pixels to tiles.
                view = new
                {
                    px_per_tile = 16f * Main.GameViewMatrix.Zoom.X * _shotScale,
                    shot_width = _shotW,
                    shot_height = _shotH,
                },
            };

            // ---- Referee-only ground truth: scoring and failure clips. NEVER shown to any model. ----
            var totals = new Dictionary<string, int>();
            for (int i = 0; i < 50; i++)
            {
                Item it = p.inventory[i];
                if (it == null || it.IsAir || string.IsNullOrEmpty(it.Name))
                    continue;
                totals[it.Name] = totals.GetValueOrDefault(it.Name) + it.stack;
            }
            NPC boss = FindBoss();
            int deaths = p.TryGetModPlayer<RLAgentPlayer>(out var modPlayer) ? modPlayer.DeathCount : 0;
            var referee = new
            {
                x = (int)(p.Center.X / 16f),
                y = (int)(p.Center.Y / 16f),
                depth_below_surface = (int)(p.Center.Y / 16f - Main.worldSurface),
                deaths,
                dayTime = Main.dayTime,
                time = Math.Round(Main.time, 0),
                defense = (int)p.statDefense,
                boss_active = boss != null,
                boss_name = boss?.GivenOrTypeName ?? "",
                boss_hp = boss?.life ?? 0,
                downed_eye = NPC.downedBoss1,
                inventory = totals,
            };

            return JsonSerializer.Serialize(new
            {
                protocol = 2,
                tick = TickCounter,
                screenshot_b64 = screenshotB64,
                screenshot_error = screenshotError,
                hud = hudBlock,
                referee,
            });
        }

        /// <summary>
        /// An enemy counts as visible only if its hitbox is inside the screen and the tile it stands in is lit.
        /// Enemies in the dark or off screen do not exist for the agent.
        /// </summary>
        internal static bool IsVisibleEnemy(NPC n)
        {
            if (n == null || !n.active || n.friendly || n.townNPC || n.life <= 0 || n.lifeMax <= 5 || n.hide || n.alpha >= 250)
                return false;
            var screen = new Rectangle((int)Main.screenPosition.X, (int)Main.screenPosition.Y, Main.screenWidth, Main.screenHeight);
            if (!screen.Intersects(n.Hitbox))
                return false;
            int tx = (int)(n.Center.X / 16f);
            int ty = (int)(n.Center.Y / 16f);
            if (tx < 0 || ty < 0 || tx >= Main.maxTilesX || ty >= Main.maxTilesY)
                return false;
            return Lighting.Brightness(tx, ty) > 0.12f;
        }

        internal static NPC NearestVisibleEnemy(Player p)
        {
            NPC best = null;
            float bestDist = float.MaxValue;
            for (int i = 0; i < Main.maxNPCs; i++)
            {
                NPC n = Main.npc[i];
                if (!IsVisibleEnemy(n))
                    continue;
                float d = Vector2.DistanceSquared(n.Center, p.Center);
                if (d < bestDist)
                {
                    bestDist = d;
                    best = n;
                }
            }
            return best;
        }

        internal static NPC FindBoss()
        {
            for (int i = 0; i < Main.maxNPCs; i++)
            {
                NPC n = Main.npc[i];
                if (n.active && n.boss)
                    return n;
            }
            return null;
        }

        // ------------------------------------------------------------------ on-screen provenance HUD

        public override void ModifyInterfaceLayers(List<GameInterfaceLayer> layers)
        {
            int idx = layers.FindIndex(l => l.Name.Equals("Vanilla: Mouse Text"));
            if (idx != -1)
            {
                layers.Insert(idx, new LegacyGameInterfaceLayer(
                    "TerrariaRLBridge: AgentHUD",
                    () =>
                    {
                        DrawAgentHud(Main.spriteBatch);
                        return true;
                    },
                    InterfaceScaleType.UI));
            }
        }

        private static string Clip(string text, int max)
        {
            if (string.IsNullOrEmpty(text)) return "";
            return text.Length > max ? text.Substring(0, max - 3) + "..." : text;
        }

        /// <summary>
        /// Shows who is really driving. Every line comes from the bridge reply; if no reply arrived, it says so.
        /// Drawn along the bottom edge so it never covers the inventory or crafting UI.
        /// </summary>
        private void DrawAgentHud(SpriteBatch sb)
        {
            if (sb == null || Main.gameMenu || Main.LocalPlayer == null || !Main.LocalPlayer.active)
                return;

            var snap = LatestControls;
            double age = snap == null ? -1 : (DateTime.UtcNow - _lastReplyUtc).TotalSeconds;
            bool online = snap != null && age <= 30.0;
            string err = _bridgeError;

            string line1;
            string line2;
            string line3;
            string line4;
            if (!online)
            {
                line1 = "AGENT BRIDGE OFFLINE - no model is controlling the player (manual play)";
                line2 = string.IsNullOrEmpty(err) ? "waiting for the Python bridge on 127.0.0.1:8765" : err;
                line3 = "SUPERVISOR: not connected | VISION: not connected | DECISION: not connected";
                line4 = "";
            }
            else
            {
                string wait = _inFlight ? "thinking" : "idle";
                line1 = $"AGENT (human-parity: sees this screen only) | reply #{_replies} {age:0.0}s ago | {wait}{(string.IsNullOrEmpty(err) ? "" : " | " + err)}";
                line2 = "SUPERVISOR " + Clip(snap.SupervisorLine, 104);
                line3 = "VISION     " + Clip(snap.VisionLine, 104);
                line4 = "DECISION   " + Clip(snap.DecisionLine, 104);
            }

            Texture2D px = TextureAssets.MagicPixel.Value;
            int boxH = 92;
            Rectangle box = new Rectangle(12, Main.screenHeight - boxH - 10, 760, boxH);
            Color accent = online ? new Color(56, 189, 248) : new Color(248, 113, 113);
            sb.Draw(px, box, new Color(8, 13, 24, 190));
            sb.Draw(px, new Rectangle(box.X, box.Y, box.Width, 2), accent);

            Utils.DrawBorderString(sb, line1, new Vector2(box.X + 8, box.Y + 6), accent, 0.72f);
            Utils.DrawBorderString(sb, line2, new Vector2(box.X + 8, box.Y + 27), Color.White, 0.68f);
            Utils.DrawBorderString(sb, line3, new Vector2(box.X + 8, box.Y + 48), new Color(134, 239, 172), 0.68f);
            Utils.DrawBorderString(sb, line4, new Vector2(box.X + 8, box.Y + 69), new Color(125, 211, 252), 0.68f);
        }
    }
}
