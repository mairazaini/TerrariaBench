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
    //   * Skills:   short closed-loop programs (Skills.cs) that press those same keys 60 times a second: walk there,
    //               chop that tree, mine that ore, fight that boss. They are input macros, not engine shortcuts.
    //   * No cheats: the mod never edits tiles, spawns items or NPCs, changes time, heals, pulls items or moves the
    //               player. Every world change goes through the same game systems a human player uses, at the same
    //               speed: real swing timing, real pickaxe power, real projectiles.
    // Absolute world coordinates and scoring ground truth are sent only inside the
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
        public string SelectItem = "";     // select item name or tool category across all 50 slots
        public int SkillId;                // id of the skill the agent wants running (0 = none); see Skills.cs
        public string SkillName = "";
        public Dictionary<string, string> SkillArgs = new Dictionary<string, string>();
        public bool LootAll;               // press "Loot All" on the chest that is currently open
        public bool AimAssist;             // legacy field; combat/tool aim assist is contextual during active use
        public string SupervisorLine = "";
        public string VisionLine = "";
        public string DecisionLine = "";
        public int ExpireTick;
        public bool OneShotsDone;
        public DateTime ReceivedAtUtc = DateTime.MinValue;
    }

    /// <summary>
    /// Presses the keys the decision model asked for and provides contextual Smart Cursor / tool-aware auto-aim
    /// while actively using a tool, weapon, or building material so clicks land cleanly without hijacking idle state.
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
            // A running skill owns the keyboard and mouse until it ends, even if the agent is slow to answer.
            SkillRunner.Sync(snap, Player);
            if (SkillRunner.Active)
            {
                SkillRunner.Tick(Player);
                return;
            }
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

            if (!string.IsNullOrEmpty(snap.SelectItem) && Player.itemAnimation == 0 && Player.itemTime == 0)
                MeteredBridgeSystem.AutoSelectItem(Player, snap.SelectItem);

            // Hotbar number key: if a slot change is requested, finish any active swing first so autoReuse items cannot lock out switching.
            bool wantSlotChange = snap.Hotbar >= 1 && snap.Hotbar <= 10 && snap.Hotbar - 1 != Player.selectedItem;
            if (wantSlotChange && Player.itemAnimation == 0 && Player.itemTime == 0)
            {
                Player.selectedItem = snap.Hotbar - 1;
                wantSlotChange = false;
            }

            if (held && snap.UseItem && !wantSlotChange)
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
                if (wantSlotChange)
                    Player.releaseUseItem = true;
                _wasUsingItem = false;
            }

            // Contextual Smart Cursor & non-interfering tool/weapon/block auto-aim while actively using an item.
            Main.SmartCursorWanted_Mouse = snap.SmartCursor;
            Vector2 aimWorld = Player.Center + new Vector2(snap.CursorDx * 16f, snap.CursorDy * 16f);
            Item heldItem = Player.HeldItem;
            if (held && snap.UseItem && !wantSlotChange && heldItem != null && !heldItem.IsAir)
            {
                if (heldItem.axe > 0)
                {
                    Main.SmartCursorWanted_Mouse = true;
                    Point? tree = MeteredBridgeSystem.FindNearestTreeBase(Player, Math.Sign(snap.CursorDx));
                    if (tree.HasValue)
                    {
                        aimWorld = new Vector2(tree.Value.X * 16f + 8f, tree.Value.Y * 16f + 8f);
                        float dxTiles = (aimWorld.X - Player.Center.X) / 16f;
                        if (Math.Abs(dxTiles) > 2.8f && Math.Abs(dxTiles) <= 14f)
                        {
                            Player.controlLeft = dxTiles < 0f;
                            Player.controlRight = dxTiles > 0f;
                            if (Collision.SolidCollision(Player.position + new Vector2(dxTiles < 0f ? -8f : Player.width, 4f), 8, Math.Max(8, Player.height - 10)))
                                Player.controlJump = true;
                        }
                    }
                }
                else if (heldItem.pick > 0)
                {
                    Point? mineTarget = MeteredBridgeSystem.FindPickaxeTarget(Player, aimWorld, snap.CursorDx, snap.CursorDy);
                    if (mineTarget.HasValue)
                        aimWorld = new Vector2(mineTarget.Value.X * 16f + 8f, mineTarget.Value.Y * 16f + 8f);
                    Main.SmartCursorWanted_Mouse = snap.SmartCursor || mineTarget.HasValue;
                }
                else if (heldItem.damage > 0 && heldItem.pick == 0 && heldItem.axe == 0 && heldItem.hammer == 0 && heldItem.createTile < 0 && heldItem.createWall < 0)
                {
                    Main.SmartCursorWanted_Mouse = false;
                    NPC enemy = MeteredBridgeSystem.NearestVisibleEnemy(Player);
                    float maxRange = (heldItem.useAmmo > 0 || heldItem.shoot > ProjectileID.None) ? 480f : 180f;
                    if (enemy != null && Vector2.DistanceSquared(enemy.Center, Player.Center) <= maxRange * maxRange)
                    {
                        aimWorld = enemy.Center + (heldItem.shoot > ProjectileID.None ? enemy.velocity * 5f : Vector2.Zero);
                        Player.direction = enemy.Center.X >= Player.Center.X ? 1 : -1;
                    }
                }
                else if (heldItem.createTile > -1 && Main.tileSolid[heldItem.createTile])
                {
                    Main.SmartCursorWanted_Mouse = false;
                    Point? placeTile = MeteredBridgeSystem.FindValidBlockPlacement(Player, aimWorld, snap.CursorDy);
                    if (placeTile.HasValue)
                        aimWorld = new Vector2(placeTile.Value.X * 16f + 8f, placeTile.Value.Y * 16f + 8f);
                }
                else if (heldItem.createWall > 0)
                {
                    Point? wallTile = MeteredBridgeSystem.FindValidWallPlacement(Player, aimWorld);
                    if (wallTile.HasValue)
                        aimWorld = new Vector2(wallTile.Value.X * 16f + 8f, wallTile.Value.Y * 16f + 8f);
                }
            }
            Main.mouseX = (int)Math.Clamp(aimWorld.X - Main.screenPosition.X, 4, Main.screenWidth - 4);
            Main.mouseY = (int)Math.Clamp(aimWorld.Y - Main.screenPosition.Y, 4, Main.screenHeight - 4);
            Terraria.GameInput.PlayerInput.MouseX = Main.mouseX;
            Terraria.GameInput.PlayerInput.MouseY = Main.mouseY;
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
        private static Point? _houseAnchor;

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
            _houseAnchor = null;
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
            _houseAnchor = null;
            SkillRunner.Reset();

            // Keep simulating and rendering when the window is not focused so unattended runs do not stall.
            if (!Main.dedServ && Main.instance != null)
                Main.instance.InactiveSleepTime = TimeSpan.Zero;
        }

        public override void OnWorldUnload()
        {
            _houseAnchor = null;
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

            if (!string.IsNullOrEmpty(snap.SelectItem))
                AutoSelectItem(p, snap.SelectItem);
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
        /// Crafts one batch through the game's own recipe list (auto-opening inventory if needed so craft clicks work).
        /// </summary>
        internal bool TryCraft(Player p, string itemName)
        {
            if (!Main.playerInventory)
            {
                Main.playerInventory = true;
                SoundEngine.PlaySound(SoundID.MenuOpen);
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
                return true;
            }

            AddEvent($"craft '{itemName}' failed: recipe not available (missing materials or crafting station)");
            return false;
        }

        /// <summary>
        /// Inventory click: armor and accessories are equipped, anything else is swapped into the selected hotbar slot.
        /// </summary>
        internal void TryUseInventoryItem(Player p, string itemName)
        {
            if (!Main.playerInventory)
            {
                Main.playerInventory = true;
                SoundEngine.PlaySound(SoundID.MenuOpen);
                Recipe.FindRecipes();
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

        internal static bool ItemMatchesQuery(Item it, string query)
        {
            if (it == null || it.IsAir || it.stack <= 0 || string.IsNullOrEmpty(query))
                return false;
            string q = query.Trim().ToLowerInvariant();
            if (string.Equals(it.Name, query.Trim(), StringComparison.OrdinalIgnoreCase))
                return true;
            return q switch
            {
                "axe" => it.axe > 0,
                "pickaxe" => it.pick > 0,
                "weapon" => it.damage > 0 && it.pick == 0 && it.axe == 0 && it.hammer == 0 && it.createTile < 0 && it.createWall < 0,
                "block" => it.createTile >= 0 && it.createTile < Main.tileSolid.Length && Main.tileSolid[it.createTile] && !Main.tileSolidTop[it.createTile],
                "platform" => it.createTile == TileID.Platforms,
                "wall" => it.createWall > 0,
                "torch" => it.createTile == TileID.Torches,
                "workbench" => it.createTile == TileID.WorkBenches,
                "chair" => it.createTile == TileID.Chairs,
                "door" => it.createTile == TileID.ClosedDoor,
                _ => it.Name.Contains(q, StringComparison.OrdinalIgnoreCase),
            };
        }

        internal static bool AutoSelectItem(Player p, string query)
        {
            if (string.IsNullOrWhiteSpace(query))
                return false;
            if (ItemMatchesQuery(p.HeldItem, query))
                return true;
            for (int i = 0; i < 10; i++)
            {
                if (string.Equals(p.inventory[i]?.Name, query.Trim(), StringComparison.OrdinalIgnoreCase) && p.inventory[i].stack > 0)
                {
                    p.selectedItem = i;
                    return true;
                }
            }
            for (int i = 0; i < 10; i++)
            {
                if (ItemMatchesQuery(p.inventory[i], query))
                {
                    p.selectedItem = i;
                    return true;
                }
            }
            for (int i = 10; i < 50; i++)
            {
                if (ItemMatchesQuery(p.inventory[i], query))
                {
                    Utils.Swap(ref p.inventory[i], ref p.inventory[9]);
                    p.selectedItem = 9;
                    return true;
                }
            }
            return false;
        }

        internal static int FindInventorySlot(Player p, string query)
        {
            for (int i = 0; i < 50; i++)
            {
                if (string.Equals(p.inventory[i]?.Name, query, StringComparison.OrdinalIgnoreCase) && p.inventory[i].stack > 0)
                    return i;
            }
            for (int i = 0; i < 50; i++)
            {
                if (ItemMatchesQuery(p.inventory[i], query))
                    return i;
            }
            return -1;
        }

        internal static bool IsTreeTile(Tile t)
        {
            if (!t.HasTile)
                return false;
            return TileID.Sets.IsATreeTrunk[t.TileType] || t.TileType == TileID.Trees || t.TileType == TileID.PalmTree;
        }

        internal static Point? FindNearestTreeBase(Player p, int preferDir)
        {
            int px = (int)(p.Center.X / 16f);
            int py = (int)(p.Center.Y / 16f);
            Point? bestPreferred = null;
            double bestPreferredDist = double.MaxValue;
            Point? bestAny = null;
            double bestAnyDist = double.MaxValue;

            for (int dx = -18; dx <= 18; dx++)
            {
                int tx = px + dx;
                if (tx <= 5 || tx >= Main.maxTilesX - 5)
                    continue;
                for (int dy = -8; dy <= 8; dy++)
                {
                    int ty = py + dy;
                    if (ty <= 5 || ty >= Main.maxTilesY - 5)
                        continue;
                    if (!IsTreeTile(Framing.GetTileSafely(tx, ty)))
                        continue;
                    int baseY = ty;
                    for (int step = 0; step < 12 && IsTreeTile(Framing.GetTileSafely(tx, baseY + 1)); step++)
                        baseY++;
                    double d2 = dx * dx + (baseY - py) * (baseY - py);
                    if (d2 < bestAnyDist)
                    {
                        bestAnyDist = d2;
                        bestAny = new Point(tx, baseY);
                    }
                    if (preferDir != 0 && Math.Sign(dx) == preferDir && d2 < bestPreferredDist)
                    {
                        bestPreferredDist = d2;
                        bestPreferred = new Point(tx, baseY);
                    }
                }
            }
            return (bestPreferred.HasValue && bestPreferredDist <= 144.0) ? bestPreferred : bestAny;
        }

        private static bool IsProtectedHouseTile(Tile t)
        {
            return t.HasTile && (
                t.TileType == TileID.WoodBlock ||
                t.TileType == TileID.ClosedDoor ||
                t.TileType == TileID.OpenDoor ||
                t.TileType == TileID.WorkBenches ||
                t.TileType == TileID.Chairs ||
                t.TileType == TileID.Torches ||
                t.TileType == TileID.Platforms ||
                t.TileType == TileID.Tables
            );
        }

        internal static Point? FindPickaxeTarget(Player p, Vector2 aimWorld, float cursorDx, float cursorDy)
        {
            int px = (int)(p.Center.X / 16f);
            int py = (int)(p.Center.Y / 16f);
            int ax = Math.Clamp((int)(aimWorld.X / 16f), 5, Main.maxTilesX - 5);
            int ay = Math.Clamp((int)(aimWorld.Y / 16f), 5, Main.maxTilesY - 5);
            bool underground = p.ZoneDirtLayerHeight || p.ZoneRockLayerHeight;

            // Priority 1: clear annoying Tombstones within reach first.
            Point? best = null;
            double bestScore = double.MaxValue;
            for (int dx = -4; dx <= 4; dx++)
            {
                for (int dy = -3; dy <= 3; dy++)
                {
                    int tx = px + dx, ty = py + dy;
                    Tile t = Framing.GetTileSafely(tx, ty);
                    if (t.HasTile && t.TileType == TileID.Tombstones)
                    {
                        double d = dx * dx + dy * dy;
                        if (d < bestScore) { bestScore = d; best = new Point(tx, ty); }
                    }
                }
            }
            if (best.HasValue)
                return best;

            // Priority 2: keep the aimed tile if solid & in reach (protect surface house structure/furniture).
            Tile aimTile = Framing.GetTileSafely(ax, ay);
            bool protectAim = !underground && IsProtectedHouseTile(aimTile)
                && !(Math.Abs(ax - px) <= 1 && ay >= py - 1 && ay <= py + 1 && Main.tileSolid[aimTile.TileType]);
            if (aimTile.HasTile && !protectAim
                && (Main.tileSolid[aimTile.TileType] || TileID.Sets.Ore[aimTile.TileType] || aimTile.TileType == TileID.Heart)
                && (ax - px) * (ax - px) + (ay - py) * (ay - py) <= 25)
            {
                if (!(cursorDy < 1.5f && Math.Abs(ax - px) <= 1 && ay >= py + 2))
                    return new Point(ax, ay);
            }

            // Priority 3: snap onto nearby ore/heart/blocking tile near the aim direction.
            best = null;
            bestScore = double.MaxValue;
            for (int dx = -4; dx <= 4; dx++)
            {
                for (int dy = -3; dy <= 3; dy++)
                {
                    if (cursorDy < 1.5f && Math.Abs(dx) <= 1 && dy >= 2)
                        continue;
                    if (cursorDx > 0.5f && dx < -1)
                        continue;
                    if (cursorDx < -0.5f && dx > 1)
                        continue;
                    int tx = px + dx, ty = py + dy;
                    Tile t = Framing.GetTileSafely(tx, ty);
                    if (!t.HasTile || Main.tileCut[t.TileType] || IsTreeTile(t))
                        continue;
                    if (!underground && IsProtectedHouseTile(t))
                        continue;
                    if (!Main.tileSolid[t.TileType] && t.TileType != TileID.Heart && !TileID.Sets.Ore[t.TileType])
                        continue;
                    double d = (tx - ax) * (tx - ax) + (ty - ay) * (ty - ay) * 1.2;
                    if (TileID.Sets.Ore[t.TileType] || t.TileType == TileID.Heart)
                        d -= 12.0;
                    if (d < bestScore) { bestScore = d; best = new Point(tx, ty); }
                }
            }
            return best;
        }

        private static bool HasPlacementSupport(int tx, int ty)
        {
            if (Framing.GetTileSafely(tx, ty).WallType > 0)
                return true;
            foreach (var (ox, oy) in new (int, int)[] { (-1, 0), (1, 0), (0, -1), (0, 1) })
            {
                Tile nt = Framing.GetTileSafely(tx + ox, ty + oy);
                if (nt.HasTile && !Main.tileCut[nt.TileType])
                    return true;
            }
            return false;
        }

        internal static Point? FindValidBlockPlacement(Player p, Vector2 aimWorld, float cursorDy)
        {
            int px = (int)(p.Center.X / 16f);
            int floorY = (int)((p.Bottom.Y + 4f) / 16f);
            int ax = Math.Clamp((int)(aimWorld.X / 16f), 5, Main.maxTilesX - 5);
            int ay = Math.Clamp((int)(aimWorld.Y / 16f), 5, Main.maxTilesY - 5);
            Point? best = null;
            double bestDist = double.MaxValue;
            // Tight -1..1 snap radius so manual clicks never fill interior air and trap the player.
            for (int rx = -1; rx <= 1; rx++)
            {
                for (int ry = -1; ry <= 1; ry++)
                {
                    int tx = ax + rx, ty = ay + ry;
                    if (cursorDy < 1.5f && Math.Abs(tx - px) <= 2 && ty >= floorY - 4 && ty <= floorY - 1)
                        continue;
                    Vector2 cellCenter = new Vector2(tx * 16f + 8f, ty * 16f + 8f);
                    if (Vector2.DistanceSquared(cellCenter, p.Center) > 84f * 84f)
                        continue;
                    if (new Rectangle(tx * 16, ty * 16, 16, 16).Intersects(p.Hitbox))
                        continue;
                    Tile t = Framing.GetTileSafely(tx, ty);
                    if (t.HasTile && !Main.tileCut[t.TileType])
                        continue;
                    if (!HasPlacementSupport(tx, ty))
                        continue;
                    double d = rx * rx + ry * ry;
                    if (d < bestDist) { bestDist = d; best = new Point(tx, ty); }
                }
            }
            return best;
        }

        internal static Point? FindValidWallPlacement(Player p, Vector2 aimWorld)
        {
            int ax = Math.Clamp((int)(aimWorld.X / 16f), 5, Main.maxTilesX - 5);
            int ay = Math.Clamp((int)(aimWorld.Y / 16f), 5, Main.maxTilesY - 5);
            Point? best = null;
            double bestDist = double.MaxValue;
            for (int rx = -3; rx <= 3; rx++)
            {
                for (int ry = -3; ry <= 3; ry++)
                {
                    int tx = ax + rx, ty = ay + ry;
                    Vector2 cellCenter = new Vector2(tx * 16f + 8f, ty * 16f + 8f);
                    if (Vector2.DistanceSquared(cellCenter, p.Center) > 84f * 84f)
                        continue;
                    Tile t = Framing.GetTileSafely(tx, ty);
                    if (t.WallType > 0)
                        continue;
                    bool supported = t.HasTile;
                    if (!supported)
                    {
                        foreach (var (ox, oy) in new (int, int)[] { (-1, 0), (1, 0), (0, -1), (0, 1) })
                        {
                            Tile nt = Framing.GetTileSafely(tx + ox, ty + oy);
                            if (nt.WallType > 0 || (nt.HasTile && Main.tileSolid[nt.TileType]))
                            {
                                supported = true;
                                break;
                            }
                        }
                    }
                    if (!supported)
                        continue;
                    double d = rx * rx + ry * ry;
                    if (d < bestDist) { bestDist = d; best = new Point(tx, ty); }
                }
            }
            return best;
        }

        private static bool IsInteriorFurnitureTile(ushort tileType)
        {
            return tileType == TileID.WorkBenches
                || tileType == TileID.Chairs
                || tileType == TileID.Torches
                || tileType == TileID.Tables
                || tileType == TileID.Furnaces
                || tileType == TileID.Anvils
                || tileType == TileID.Platforms;
        }

        private static Point? FindExistingHouseAnchor(int px, int floorY)
        {
            int minWx = int.MaxValue, maxWx = int.MinValue, maxWy = int.MinValue;
            int wallCount = 0;
            Point? benchPt = null;
            for (int dx = -24; dx <= 24; dx++)
            {
                int tx = px + dx;
                if (tx <= 6 || tx >= Main.maxTilesX - 6) continue;
                for (int ty = Math.Max(6, floorY - 8); ty <= Math.Min(Main.maxTilesY - 6, floorY + 3); ty++)
                {
                    Tile t = Framing.GetTileSafely(tx, ty);
                    if (t.WallType > 0 && Main.wallHouse[t.WallType])
                    {
                        wallCount++;
                        if (tx < minWx) minWx = tx;
                        if (tx > maxWx) maxWx = tx;
                        if (ty > maxWy) maxWy = ty;
                    }
                    if (!benchPt.HasValue && t.HasTile && t.TileType == TileID.WorkBenches)
                        benchPt = new Point(tx, ty + 1);
                }
            }
            if (wallCount >= 8 && maxWx >= minWx)
            {
                int centerX = Math.Clamp((minWx + maxWx) / 2, 10, Main.maxTilesX - 16);
                int roomFloorY = Math.Clamp(maxWy + 1, 12, Main.maxTilesY - 6);
                return new Point(centerX, roomFloorY);
            }
            return benchPt;
        }

        private static bool HasTileInRect(int minX, int maxX, int minY, int maxY, ushort tileTypeA, ushort? tileTypeB = null)
        {
            for (int tx = Math.Max(5, minX); tx <= Math.Min(Main.maxTilesX - 5, maxX); tx++)
            {
                for (int ty = Math.Max(5, minY); ty <= Math.Min(Main.maxTilesY - 5, maxY); ty++)
                {
                    Tile t = Framing.GetTileSafely(tx, ty);
                    if (!t.HasTile) continue;
                    if (t.TileType == tileTypeA || (tileTypeB.HasValue && t.TileType == tileTypeB.Value))
                        return true;
                }
            }
            return false;
        }

        private static bool HasNearbyTile(int cx, int cy, int r, ushort tileType)
        {
            for (int dx = -r; dx <= r; dx++)
                for (int dy = -r; dy <= r; dy++)
                {
                    Tile t = Framing.GetTileSafely(cx + dx, cy + dy);
                    if (t.HasTile && t.TileType == tileType)
                        return true;
                }
            return false;
        }

        internal void AddEvent(string text)
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
                        SelectItem = GetStr(ctrl, "select_item", ""),
                        LootAll = GetBool(ctrl, "loot_all"),
                        AimAssist = GetBool(ctrl, "aim_assist"),
                        ExpireTick = TickCounter + hold,
                        ReceivedAtUtc = DateTime.UtcNow,
                    };
                    if (ctrl.TryGetProperty("skill", out JsonElement skill) && skill.ValueKind == JsonValueKind.Object)
                    {
                        snap.SkillId = GetInt(skill, "id", 0);
                        snap.SkillName = GetStr(skill, "name", "");
                        if (skill.TryGetProperty("args", out JsonElement args) && args.ValueKind == JsonValueKind.Object)
                        {
                            foreach (JsonProperty arg in args.EnumerateObject())
                            {
                                snap.SkillArgs[arg.Name] = arg.Value.ValueKind == JsonValueKind.String
                                    ? (arg.Value.GetString() ?? "")
                                    : arg.Value.GetRawText();
                            }
                        }
                    }
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
            var backpack = new Dictionary<string, int>();
            for (int i = 0; i < 50; i++)
            {
                Item it = p.inventory[i];
                if (it == null || it.IsAir || string.IsNullOrEmpty(it.Name))
                    continue;
                backpack[it.Name] = backpack.GetValueOrDefault(it.Name) + it.stack;
            }
            var equipmentNow = new List<string>();
            for (int i = 0; i < 8 && i < p.armor.Length; i++)
            {
                Item it = p.armor[i];
                if (it != null && !it.IsAir)
                    equipmentNow.Add(it.Name);
            }
            Recipe.FindRecipes();
            var craftableNow = new List<object>();
            var seenNow = new HashSet<string>();
            for (int i = 0; i < Main.numAvailableRecipes && craftableNow.Count < 40; i++)
            {
                Recipe r = Main.recipe[Main.availableRecipe[i]];
                if (r?.createItem == null || r.createItem.IsAir || !seenNow.Add(r.createItem.Name))
                    continue;
                craftableNow.Add(new { name = r.createItem.Name, makes = r.createItem.stack });
            }

            Dictionary<string, int> inventory = null;
            List<string> equipment = null;
            List<object> craftable = null;
            if (invOpen)
            {
                // Only visible while the inventory screen is open, preserving existing UI fields.
                inventory = new Dictionary<string, int>();
                for (int i = 10; i < 50; i++)
                {
                    Item it = p.inventory[i];
                    if (it == null || it.IsAir || string.IsNullOrEmpty(it.Name))
                        continue;
                    inventory[it.Name] = inventory.GetValueOrDefault(it.Name) + it.stack;
                }
                equipment = new List<string>(equipmentNow);
                craftable = new List<object>(craftableNow);
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

            var visibleEnemies = new List<object>();
            for (int i = 0; i < Main.maxNPCs && visibleEnemies.Count < 8; i++)
            {
                NPC n = Main.npc[i];
                if (!IsVisibleEnemy(n))
                    continue;
                float dx = (n.Center.X - p.Center.X) / 16f;
                float dy = (n.Center.Y - p.Center.Y) / 16f;
                float dist = (float)Math.Sqrt(dx * dx + dy * dy);
                bool approaching = (dx > 0 && n.velocity.X < -0.1f) || (dx < 0 && n.velocity.X > 0.1f);
                bool lineOfSight = Collision.CanHitLine(p.position, p.width, p.height, n.position, n.width, n.height);
                visibleEnemies.Add(new
                {
                    name = n.GivenOrTypeName,
                    dx = Math.Round(dx, 1),
                    dy = Math.Round(dy, 1),
                    dist = Math.Round(dist, 1),
                    hp = n.life,
                    max_hp = n.lifeMax,
                    approaching,
                    line_of_sight = lineOfSight,
                });
            }

            var worldView = BuildWorldView(p);

            var blocked = new
            {
                left = Collision.SolidCollision(p.position + new Vector2(-8f, 4f), 8, Math.Max(8, p.height - 8)),
                right = Collision.SolidCollision(p.position + new Vector2(p.width, 4f), 8, Math.Max(8, p.height - 8)),
                above = Collision.SolidCollision(p.position + new Vector2(2f, -6f), Math.Max(8, p.width - 4), 6),
                below = Collision.SolidCollision(p.position + new Vector2(2f, p.height), Math.Max(8, p.width - 4), 8),
            };

            var snap = LatestControls;
            NPC boss = FindBoss();
            double clock = Main.dayTime ? 4.5 + Main.time / 3600.0 : (19.5 + Main.time / 3600.0) % 24.0;
            var hudBlock = new
            {
                // Skill-runner status and the facts a planner needs that a player reads off the map, the
                // equipment page, the boss health bar and a watch.
                skill = SkillRunner.Hud(),
                pos = new { x = (int)(p.Center.X / 16f) - Main.spawnTileX, y = (int)(p.Center.Y / 16f) - Main.spawnTileY },
                defense = (int)p.statDefense,
                clock = Math.Round(clock, 2),
                evil_biome = p.ZoneCorrupt || p.ZoneCrimson,
                world = Main.worldName,
                boss = boss == null ? null : new
                {
                    name = boss.GivenOrTypeName,
                    hp = boss.life,
                    max_hp = boss.lifeMax,
                    dx = Math.Round((boss.Center.X - p.Center.X) / 16f, 1),
                    dy = Math.Round((boss.Center.Y - p.Center.Y) / 16f, 1),
                },
                hp = p.statLife,
                max_hp = p.statLifeMax2,
                mana = p.statMana,
                max_mana = p.statManaMax2,
                dead = p.dead,
                time_phase = Main.dayTime ? "Day" : "Night",
                blocked,
                selected_slot = p.selectedItem + 1,
                held_item = p.HeldItem == null || p.HeldItem.IsAir ? "" : p.HeldItem.Name,
                hotbar,
                backpack,
                equipment_now = equipmentNow,
                craftable_now = craftableNow,
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
                world_view = worldView,
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

        internal static string OreDisplayName(ushort tileType)
        {
            return tileType switch
            {
                TileID.Copper => "Copper Ore",
                TileID.Tin => "Tin Ore",
                TileID.Iron => "Iron Ore",
                TileID.Lead => "Lead Ore",
                TileID.Silver => "Silver Ore",
                TileID.Tungsten => "Tungsten Ore",
                TileID.Gold => "Gold Ore",
                TileID.Platinum => "Platinum Ore",
                TileID.Demonite => "Demonite Ore",
                TileID.Crimtane => "Crimtane Ore",
                _ => TileID.Search.GetName(tileType) ?? $"Ore#{tileType}",
            };
        }

        private static object BuildWorldView(Player p)
        {
            int px = Math.Clamp((int)(p.Center.X / 16f), 5, Main.maxTilesX - 5);
            int py = Math.Clamp((int)(p.Center.Y / 16f), 5, Main.maxTilesY - 5);
            int floorY = Math.Clamp((int)((p.Bottom.Y + 4f) / 16f), 5, Main.maxTilesY - 5);

            float light = Math.Max(Lighting.Brightness(px, py), Lighting.Brightness(px, Math.Max(1, py - 1)));
            string lighting = light >= 0.45f ? "bright" : (light >= 0.18f ? "dim" : (light >= 0.05f ? "dark" : "pitch_black"));
            string biome = p.ZoneRockLayerHeight ? "cavern" : (p.ZoneDirtLayerHeight ? "underground" : "surface");

            var trees = new List<(double dist, object item)>();
            var treeCols = new HashSet<int>();
            var ores = new List<(double dist, object item)>();
            var chests = new List<(double dist, object item)>();
            var hearts = new List<(double dist, object item)>();
            var stations = new List<(double dist, object item)>();
            int tombstonesNearby = 0;

            for (int dx = -28; dx <= 28; dx++)
            {
                int tx = px + dx;
                if (tx <= 5 || tx >= Main.maxTilesX - 5) continue;
                for (int dy = -18; dy <= 18; dy++)
                {
                    int ty = py + dy;
                    if (ty <= 5 || ty >= Main.maxTilesY - 5) continue;
                    Tile t = Framing.GetTileSafely(tx, ty);
                    if (!t.HasTile) continue;

                    double dist = Math.Sqrt(dx * dx + dy * dy);
                    if (Math.Abs(dy) <= 14 && IsTreeTile(t) && treeCols.Add(tx))
                    {
                        int baseY = ty;
                        for (int s = 0; s < 12 && IsTreeTile(Framing.GetTileSafely(tx, baseY + 1)); s++)
                            baseY++;
                        int bdy = baseY - py;
                        trees.Add((Math.Sqrt(dx * dx + bdy * bdy), new { dx, dy = bdy, dist = Math.Round(Math.Sqrt(dx * dx + bdy * bdy), 1) }));
                    }
                    if (Math.Abs(dx) <= 24 && TileID.Sets.Ore[t.TileType])
                    {
                        ores.Add((dist, new { name = OreDisplayName(t.TileType), dx, dy, dist = Math.Round(dist, 1) }));
                    }
                    if ((t.TileType == TileID.Containers || t.TileType == TileID.Containers2) && t.TileFrameX % 36 == 0 && t.TileFrameY % 36 == 0)
                    {
                        chests.Add((dist, new { dx, dy, dist = Math.Round(dist, 1) }));
                    }
                    if (t.TileType == TileID.Heart && t.TileFrameX % 36 == 0 && t.TileFrameY % 36 == 0)
                    {
                        hearts.Add((dist, new { dx, dy, dist = Math.Round(dist, 1) }));
                    }
                    if (Math.Abs(dx) <= 15 && Math.Abs(dy) <= 15)
                    {
                        string kind = t.TileType switch
                        {
                            TileID.WorkBenches when t.TileFrameX % 36 == 0 => "Work Bench",
                            TileID.Furnaces when t.TileFrameX % 54 == 0 && t.TileFrameY == 0 => "Furnace",
                            TileID.Anvils when t.TileFrameX % 36 == 0 => "Anvil",
                            TileID.Chairs when t.TileFrameY == 0 => "Chair",
                            TileID.ClosedDoor or TileID.OpenDoor when t.TileFrameY == 0 => "Door",
                            TileID.Torches => "Torch",
                            TileID.Campfire when t.TileFrameX % 54 == 0 && t.TileFrameY == 0 => "Campfire",
                            TileID.Tombstones when t.TileFrameX % 36 == 0 && t.TileFrameY == 0 => "Tombstone",
                            _ => null,
                        };
                        if (kind != null)
                        {
                            stations.Add((dist, new { kind, dx, dy }));
                            if (kind == "Tombstone" && Math.Abs(dx) <= 8 && Math.Abs(dy) <= 5)
                                tombstonesNearby++;
                        }
                    }
                }
            }

            // Demon/Crimson Altars are what the Suspicious Looking Eye is crafted at; look a little further for them.
            var altars = new List<(double dist, object item)>();
            for (int dx = -60; dx <= 60; dx++)
            {
                int tx = px + dx;
                if (tx <= 5 || tx >= Main.maxTilesX - 5) continue;
                for (int dy = -34; dy <= 34; dy++)
                {
                    int ty = py + dy;
                    if (ty <= 5 || ty >= Main.maxTilesY - 5) continue;
                    Tile t = Framing.GetTileSafely(tx, ty);
                    if (t.HasTile && t.TileType == TileID.DemonAltar && t.TileFrameX % 54 == 0 && t.TileFrameY == 0)
                        altars.Add((Math.Sqrt(dx * dx + dy * dy), new { dx, dy }));
                }
            }
            altars.Sort((a, b) => a.dist.CompareTo(b.dist));

            trees.Sort((a, b) => a.dist.CompareTo(b.dist));
            ores.Sort((a, b) => a.dist.CompareTo(b.dist));
            chests.Sort((a, b) => a.dist.CompareTo(b.dist));
            hearts.Sort((a, b) => a.dist.CompareTo(b.dist));
            stations.Sort((a, b) => a.dist.CompareTo(b.dist));

            Point? anchor = _houseAnchor ?? FindExistingHouseAnchor(px, floorY);
            int hcx = anchor?.X ?? px;
            int hFloorY = anchor?.Y ?? floorY;
            int hLeftX = Math.Clamp(hcx - 5, 6, Main.maxTilesX - 20);
            int hRightX = hLeftX + 11;
            int hRoofY = Math.Max(6, hFloorY - 6);

            int backWalls = 0, roofTiles6 = 0, roofTiles5 = 0, floorTiles = 0;
            bool hasBench = false, hasChair = false, hasDoor = false, hasLight = false;
            for (int tx = hLeftX; tx <= hRightX; tx++)
            {
                Tile ft = Framing.GetTileSafely(tx, hFloorY);
                if (ft.HasTile && Main.tileSolid[ft.TileType]) floorTiles++;
                Tile rt6 = Framing.GetTileSafely(tx, hRoofY);
                if (rt6.HasTile && Main.tileSolid[rt6.TileType]) roofTiles6++;
                Tile rt5 = Framing.GetTileSafely(tx, hFloorY - 5);
                if (rt5.HasTile && Main.tileSolid[rt5.TileType]) roofTiles5++;

                bool insideX = tx >= hLeftX + 1 && tx <= hRightX - 1;
                for (int ty = hRoofY + 1; ty <= hFloorY - 1; ty++)
                {
                    Tile t = Framing.GetTileSafely(tx, ty);
                    if (insideX && t.WallType > 0 && Main.wallHouse[t.WallType]) backWalls++;
                    if (!t.HasTile) continue;
                    if (t.TileType == TileID.ClosedDoor || t.TileType == TileID.OpenDoor) hasDoor = true;
                    if (!insideX) continue;
                    if (t.TileType == TileID.WorkBenches) hasBench = true;
                    if (t.TileType == TileID.Chairs) hasChair = true;
                    if (t.TileType == TileID.Torches || t.TileType == TileID.Campfire) hasLight = true;
                }
            }
            int roofTiles = Math.Max(roofTiles6, roofTiles5);

            int localRoof = 0, localWalls = 0;
            for (int dx = -3; dx <= 3; dx++)
            {
                int ltx = Math.Clamp(px + dx, 5, Main.maxTilesX - 5);
                for (int dy = -6; dy <= -3; dy++)
                {
                    Tile lt = Framing.GetTileSafely(ltx, Math.Max(5, floorY + dy));
                    if (lt.HasTile && (lt.TileType == TileID.WoodBlock || Main.tileSolid[lt.TileType])) localRoof++;
                }
                for (int ty = Math.Max(5, floorY - 5); ty <= floorY - 1; ty++)
                {
                    Tile lt = Framing.GetTileSafely(ltx, ty);
                    if (lt.WallType > 0 && Main.wallHouse[lt.WallType]) localWalls++;
                }
            }

            bool validRoom = WorldGen.StartRoomCheck(px, Math.Max(5, floorY - 2))
                || (anchor.HasValue && WorldGen.StartRoomCheck(hcx, Math.Max(5, hFloorY - 2)));
            if (validRoom && !_houseAnchor.HasValue)
                _houseAnchor = anchor ?? new Point(px, floorY);
            bool inHouse = localRoof >= 4 && localWalls >= 6;
            var missingForHouse = new List<string>();
            if (!validRoom)
            {
                if (floorTiles < 12) missingForHouse.Add("floor_gaps");
                if (roofTiles < 10) missingForHouse.Add("roof_and_frame");
                if (backWalls < 40) missingForHouse.Add("back_walls");
                if (!hasDoor) missingForHouse.Add("door");
                if (!hasBench) missingForHouse.Add("workbench_inside");
                if (!hasChair) missingForHouse.Add("chair_inside");
                if (!hasLight) missingForHouse.Add("torch_inside");
                if (missingForHouse.Count == 0) missingForHouse.Add("room_enclosure_check");
            }

            var caves = new List<object>();
            for (int dx = -20; dx <= 20 && caves.Count < 3; dx += 2)
            {
                int tx = px + dx;
                if (tx <= 5 || tx >= Main.maxTilesX - 5) continue;
                int airRun = 0;
                for (int dy = 1; dy <= 14; dy++)
                {
                    Tile t = Framing.GetTileSafely(tx, py + dy);
                    if (!t.HasTile || !Main.tileSolid[t.TileType]) airRun++;
                    else break;
                }
                if (airRun >= 5)
                    caves.Add(new { dx, dy = 2 });
            }

            var asciiMap = new List<string>(13);
            for (int dy = -6; dy <= 6; dy++)
            {
                var row = new char[25];
                for (int dx = -12; dx <= 12; dx++)
                {
                    int col = dx + 12;
                    if (dx == 0 && Math.Abs(dy) <= 1)
                    {
                        row[col] = '@';
                        continue;
                    }
                    int tx = Math.Clamp(px + dx, 1, Main.maxTilesX - 2);
                    int ty = Math.Clamp(py + dy, 1, Main.maxTilesY - 2);
                    Tile t = Framing.GetTileSafely(tx, ty);
                    if (!t.HasTile)
                        row[col] = t.WallType > 0 && Main.wallHouse[t.WallType] ? ',' : '.';
                    else if (IsTreeTile(t)) row[col] = 'T';
                    else if (TileID.Sets.Ore[t.TileType]) row[col] = '*';
                    else if (t.TileType == TileID.Containers || t.TileType == TileID.Containers2) row[col] = 'C';
                    else if (t.TileType == TileID.Heart) row[col] = 'H';
                    else if (t.TileType == TileID.WorkBenches || t.TileType == TileID.Furnaces || t.TileType == TileID.Anvils) row[col] = 'W';
                    else if (t.TileType == TileID.ClosedDoor || t.TileType == TileID.OpenDoor) row[col] = 'D';
                    else if (t.TileType == TileID.Platforms) row[col] = '=';
                    else if (t.TileType == TileID.Tombstones) row[col] = 'X';
                    else if (t.TileType == TileID.Torches || t.TileType == TileID.Campfire) row[col] = '+';
                    else if (Main.tileSolid[t.TileType]) row[col] = '#';
                    else row[col] = '.';
                }
                asciiMap.Add(new string(row));
            }

            return new
            {
                lighting,
                biome,
                trees = trees.GetRange(0, Math.Min(6, trees.Count)).ConvertAll(x => x.item),
                ores = ores.GetRange(0, Math.Min(8, ores.Count)).ConvertAll(x => x.item),
                chests = chests.GetRange(0, Math.Min(4, chests.Count)).ConvertAll(x => x.item),
                life_crystals = hearts.GetRange(0, Math.Min(4, hearts.Count)).ConvertAll(x => x.item),
                stations = stations.GetRange(0, Math.Min(16, stations.Count)).ConvertAll(x => x.item),
                altars = altars.GetRange(0, Math.Min(3, altars.Count)).ConvertAll(x => x.item),
                house_status = new
                {
                    valid_room = validRoom,
                    in_house = inHouse,
                    back_walls = backWalls,
                    roof_tiles = roofTiles,
                    floor_tiles = floorTiles,
                    has_bench = hasBench,
                    has_chair = hasChair,
                    has_door = hasDoor,
                    has_light = hasLight,
                    tombstones_nearby = tombstonesNearby,
                    missing_for_house = missingForHouse,
                },
                cave_openings = caves,
                ascii_map = asciiMap,
            };
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
                if (SkillRunner.Id != 0 && !string.IsNullOrEmpty(SkillRunner.Name))
                {
                    string why = SkillRunner.Active ? SkillRunner.Detail : SkillRunner.Reason;
                    line3 = "SKILL      " + Clip($"{SkillRunner.Name} [{SkillRunner.Status}] {SkillRunner.Ticks / 60}s {why}", 104);
                }
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
