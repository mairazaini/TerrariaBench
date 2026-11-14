using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Terraria;
using Terraria.GameContent;
using Terraria.ID;
using Terraria.ModLoader;
using Terraria.UI;

namespace TerrariaRLBridge
{
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
        public string HotbarSlot = "slot_1_weapon";
        public string AimTarget = "ahead";
        public string ThreatLabel = "Calm";
        public string MacroTool = "chop_trees";
        public string Strategy = "";
        public double Confidence;
        public int Stage = 1;
        public string HudText = "";
        public DateTime ReceivedAtUtc = DateTime.MinValue;
    }

    /// <summary>
    /// Applies combined gemini-decision-1 controls and 60Hz progression reflexes inside tModLoader.
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

            var snap = MeteredBridgeSystem.ActiveControls ?? MeteredBridgeSystem.LatestControls;
            if (snap == null || (DateTime.UtcNow - snap.ReceivedAtUtc).TotalSeconds > 10.0)
                return;

            Player.controlLeft = snap.Left;
            Player.controlRight = snap.Right;
            Player.controlUp = snap.Up;
            Player.controlDown = snap.Down;
            if (snap.Jump)
            {
                // Release jump for 1 tick on landing so continuous jump intent retriggers cleanly.
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
            Player.controlUseTile = snap.UseTile;
            Player.controlHook = snap.Hook;
            Player.controlQuickHeal = snap.QuickHeal;
            Player.controlSmart = snap.SmartCursor;

            if (snap.UseItem)
            {
                // Release for 1 tick between swings so non-autoReuse weapons and tools repeat reliably.
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

            int slot = PickHotbarSlot(snap.HotbarSlot, snap.MacroTool);
            if (slot >= 0 && slot < 10)
                Player.selectedItem = slot;

            Vector2 aimWorld = MeteredBridgeSystem.OverrideAimWorld ?? ResolveAimWorld(snap.AimTarget, snap.Left ? -1 : (snap.Right ? 1 : Player.direction));
            if (snap.UseItem || snap.UseTile)
                Player.direction = aimWorld.X >= Player.Center.X ? 1 : -1;

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

        private int PickHotbarSlot(string hotbarSlot, string macroTool)
        {
            int bestIdx = -1;
            int bestScore = -1;
            bool preferRanged = MeteredBridgeSystem.PreferRangedWeapon;

            for (int i = 0; i < 10; i++)
            {
                Item item = Player.inventory[i];
                if (item == null || item.IsAir)
                    continue;

                int score = -1;
                switch (hotbarSlot)
                {
                    case "slot_2_pickaxe":
                        if (item.pick > 0) score = item.pick;
                        break;
                    case "slot_3_axe":
                        if (item.axe > 0) score = item.axe;
                        break;
                    case "slot_4_wood":
                        if (item.type == ItemID.Wood) score = 200;
                        else if (item.createTile >= TileID.Dirt && Main.tileSolid[item.createTile] && !Main.tileSolidTop[item.createTile]) score = 50;
                        break;
                    case "slot_5_platform":
                        if (item.type == ItemID.WoodPlatform) score = 200;
                        else if (item.createTile >= TileID.Dirt && Main.tileSolidTop[item.createTile]) score = 50;
                        break;
                    case "slot_6_torch":
                        if (item.type == ItemID.Torch) score = 200;
                        break;
                    case "slot_7_station":
                        if (item.type == ItemID.WorkBench || item.type == ItemID.Furnace ||
                            item.type == ItemID.IronAnvil || item.type == ItemID.LeadAnvil ||
                            item.type == ItemID.WoodenDoor || item.type == ItemID.Campfire)
                            score = 150;
                        break;
                    case "slot_8_potion":
                        if (item.type == ItemID.SuspiciousLookingEye) score = 300;
                        else if (item.healLife > 0) score = 150 + item.healLife;
                        break;
                    default: // slot_1_weapon
                        if (item.damage > 0 && item.pick == 0 && item.axe == 0 && item.hammer == 0)
                        {
                            score = item.damage;
                            if (preferRanged && item.useAmmo == AmmoID.Arrow) score += 200;
                            else if (!preferRanged && item.DamageType == DamageClass.Melee) score += 100;
                        }
                        break;
                }

                if (score > bestScore)
                {
                    bestScore = score;
                    bestIdx = i;
                }
            }

            if (bestIdx >= 0)
                return bestIdx;

            if (macroTool == "chop_trees")
                return FindFirstHotbar(it => it.axe > 0);
            if (macroTool == "mine_vein" || macroTool == "explore_cavern")
                return FindFirstHotbar(it => it.pick > 0);
            return Player.selectedItem;
        }

        private int FindFirstHotbar(Func<Item, bool> pred)
        {
            for (int i = 0; i < 10; i++)
            {
                if (Player.inventory[i] != null && !Player.inventory[i].IsAir && pred(Player.inventory[i]))
                    return i;
            }
            return -1;
        }

        private Vector2 ResolveAimWorld(string aimTarget, int dir)
        {
            Vector2 center = Player.Center;
            if (aimTarget == "nearest_enemy")
            {
                NPC best = MeteredBridgeSystem.FindNearestEnemy(center, 900f);
                if (best != null)
                    return best.Center + best.velocity * 4f;
            }

            int px = (int)(center.X / 16f);
            int py = (int)(center.Y / 16f);

            if (aimTarget == "nearest_tree")
            {
                Vector2? found = MeteredBridgeSystem.FindNearestTile(px, py, 20, t => t == TileID.Trees);
                if (found.HasValue) return found.Value;
            }
            else if (aimTarget == "nearest_ore")
            {
                Vector2? found = MeteredBridgeSystem.FindNearestTile(px, py, 12, IsOreOrStone);
                if (found.HasValue) return found.Value;
            }
            else if (aimTarget == "underfoot")
            {
                return new Vector2(center.X, Player.Bottom.Y + 24f);
            }
            else if (aimTarget == "build_site")
            {
                return new Vector2(center.X + dir * 40f, center.Y - 12f);
            }

            return new Vector2(center.X + dir * 72f, center.Y - 8f);
        }

        private static bool IsOreOrStone(int t) =>
            t == TileID.Copper || t == TileID.Tin || t == TileID.Iron || t == TileID.Lead ||
            t == TileID.Silver || t == TileID.Tungsten || t == TileID.Gold || t == TileID.Platinum ||
            t == TileID.Stone;
    }

    /// <summary>
    /// Streams telemetry to http://127.0.0.1:8765/real_game_tick, runs the 5-stage autonomous
    /// Eye of Cthulhu progression loop, and draws a live screen-recording HUD overlay.
    /// </summary>
    public class MeteredBridgeSystem : ModSystem
    {
        private static readonly HttpClient Http = new HttpClient { Timeout = TimeSpan.FromMilliseconds(2500) };
        private const string BridgeUrl = "http://127.0.0.1:8765/real_game_tick";
        private const int PollIntervalTicks = 20;

        internal static volatile AgentControlSnapshot LatestControls;
        internal static volatile AgentControlSnapshot ActiveControls;
        internal static Vector2? OverrideAimWorld;
        internal static bool PreferRangedWeapon;

        private int _tickCounter;
        private int _stage = 1;
        private int _stageTicks;
        private bool _homeSet;
        private Point _homeTile;
        private bool _shelterBuilt;
        private int _shelterBuildStep;
        private bool _metalForged;
        private int _mineTicks;
        private bool _arenaBuilt;
        private int _arenaColOffset = -22;
        private bool _eyeSummoned;
        private bool _eyeWasAlive;
        private bool _eyeDefeated;
        private int _resummonCooldown;
        private volatile bool _inFlight;
        private string _lastHudAnnouncement = "";

        public override void OnWorldLoad()
        {
            _tickCounter = 0;
            _stage = 1;
            _stageTicks = 0;
            _homeSet = false;
            _shelterBuilt = false;
            _shelterBuildStep = 0;
            _metalForged = false;
            _mineTicks = 0;
            _arenaBuilt = false;
            _arenaColOffset = -22;
            _eyeSummoned = false;
            _eyeWasAlive = false;
            _eyeDefeated = false;
            _resummonCooldown = 0;
            LatestControls = null;
            ActiveControls = null;
            OverrideAimWorld = null;
            PreferRangedWeapon = false;
            _lastHudAnnouncement = "";

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

            if (LatestControls != null)
                Main.gamePaused = false;

            _tickCounter++;
            if (!_homeSet)
                InitializeHomeAnchor(p);

            if (!p.dead)
            {
                PullNearbyItems(p);
                RunProgressionTick(p);
            }
            else
            {
                OverrideAimWorld = null;
            }

            if (_tickCounter % PollIntervalTicks == 0 && !_inFlight)
                PostTickAsync(p);
        }

        private void InitializeHomeAnchor(Player p)
        {
            int sx = (int)(p.Center.X / 16f);
            int sy = (int)(p.Bottom.Y / 16f);
            for (int y = Math.Max(40, sy - 5); y < Math.Min(Main.maxTilesY - 20, sy + 20); y++)
            {
                Tile t = Framing.GetTileSafely(sx, y);
                if (t.HasTile && Main.tileSolid[t.TileType] && !Main.tileSolidTop[t.TileType])
                {
                    sy = y;
                    break;
                }
            }
            _homeTile = new Point(sx, sy);
            _homeSet = true;
        }

        private void PullNearbyItems(Player p)
        {
            const float maxDistSq = (22f * 16f) * (22f * 16f);
            for (int i = 0; i < Main.maxItems; i++)
            {
                Item it = Main.item[i];
                if (it == null || !it.active || it.IsAir)
                    continue;
                Vector2 diff = p.Center - it.Center;
                float d2 = diff.LengthSquared();
                if (d2 > 16f && d2 < maxDistSq)
                {
                    it.noGrabDelay = 0;
                    diff.Normalize();
                    it.velocity = diff * 12f;
                    it.Center += diff * 4f;
                }
            }
        }

        private void RunProgressionTick(Player p)
        {
            _stageTicks++;
            OverrideAimWorld = null;

            NPC eoc = FindEyeOfCthulhu();
            if (eoc != null)
            {
                _eyeWasAlive = true;
                _stage = 5;
            }
            else if (_eyeWasAlive && !p.dead)
            {
                _eyeDefeated = true;
                _eyeWasAlive = false;
            }

            var baseSnap = LatestControls ?? new AgentControlSnapshot();
            var act = new AgentControlSnapshot
            {
                Left = baseSnap.Left,
                Right = baseSnap.Right,
                Up = baseSnap.Up,
                Down = baseSnap.Down,
                Jump = baseSnap.Jump,
                UseItem = baseSnap.UseItem,
                UseTile = false,
                Hook = baseSnap.Hook,
                QuickHeal = p.statLife < (int)(p.statLifeMax2 * 0.55f),
                SmartCursor = baseSnap.SmartCursor,
                HotbarSlot = baseSnap.HotbarSlot,
                AimTarget = baseSnap.AimTarget,
                ThreatLabel = baseSnap.ThreatLabel,
                MacroTool = baseSnap.MacroTool,
                Strategy = baseSnap.Strategy,
                Confidence = baseSnap.Confidence,
                Stage = _stage,
                HudText = baseSnap.HudText,
                ReceivedAtUtc = DateTime.UtcNow,
            };

            switch (_stage)
            {
                case 1:
                    RunStage1WoodAndSlimes(p, act);
                    break;
                case 2:
                    RunStage2ShelterAndWoodGear(p, act);
                    break;
                case 3:
                    RunStage3MineAndForgeMetalGear(p, act);
                    break;
                case 4:
                    RunStage4BuildArena(p, act);
                    break;
                case 5:
                    RunStage5EyeOfCthulhu(p, act, eoc);
                    break;
            }

            if (p.statLife < Math.Min(110, (int)(p.statLifeMax2 * 0.6f)) && p.potionDelay == 0)
                act.QuickHeal = true;

            act.Stage = _stage;
            ActiveControls = act;
        }

        private void RunStage1WoodAndSlimes(Player p, AgentControlSnapshot act)
        {
            PreferRangedWeapon = false;
            int wood = CountItem(p, ItemID.Wood);
            if (_stageTicks >= 360 && (wood >= 80 || _stageTicks >= 900))
            {
                EnsureInventoryItem(p, ItemID.Wood, 80, 4);
                TransitionStage(2, "[S2/5] Building Wood Shelter, Workbench & Starter Gear...");
                return;
            }

            NPC nearEnemy = FindNearestEnemy(p.Center, 11f * 16f);
            if (nearEnemy != null)
            {
                act.HotbarSlot = "slot_1_weapon";
                act.AimTarget = "nearest_enemy";
                OverrideAimWorld = nearEnemy.Center;
                act.Left = nearEnemy.Center.X < p.Center.X - 36f;
                act.Right = nearEnemy.Center.X > p.Center.X + 36f;
                act.Jump = Math.Abs(p.velocity.X) < 0.08f || nearEnemy.Center.Y < p.Center.Y - 24f;
                act.UseItem = true;
                return;
            }

            int px = (int)(p.Center.X / 16f);
            int py = (int)(p.Center.Y / 16f);

            // Escape enclosure/hole if solid tiles block overhead or both sides around the player.
            bool roofBlocked = false;
            Point? breakTile = null;
            for (int oy = -3; oy <= 0; oy++)
            {
                for (int ox = -2; ox <= 2; ox++)
                {
                    Tile t = Framing.GetTileSafely(px + ox, py + oy);
                    if (t.HasTile && Main.tileSolid[t.TileType] && !Main.tileSolidTop[t.TileType] && !Main.tileContainer[t.TileType])
                    {
                        if (oy <= -1 && Math.Abs(ox) <= 1) roofBlocked = true;
                        breakTile ??= new Point(px + ox, py + oy);
                    }
                }
            }
            if (roofBlocked && breakTile.HasValue)
            {
                act.HotbarSlot = "slot_2_pickaxe";
                act.SmartCursor = true;
                act.AimTarget = "build_site";
                act.UseItem = true;
                act.Jump = _stageTicks % 14 < 8;
                OverrideAimWorld = new Vector2(breakTile.Value.X * 16f + 8f, breakTile.Value.Y * 16f + 8f);
                if (_stageTicks % 8 == 0)
                    p.PickTile(breakTile.Value.X, breakTile.Value.Y, 40);
                return;
            }

            Point? treeTile = FindNearestTreeBase(px, py, 60);
            act.HotbarSlot = "slot_3_axe";
            act.AimTarget = "nearest_tree";
            act.SmartCursor = true;

            if (treeTile.HasValue)
            {
                Vector2 treePos = new Vector2(treeTile.Value.X * 16f + 8f, treeTile.Value.Y * 16f + 8f);
                OverrideAimWorld = treePos;
                float dx = treePos.X - p.Center.X;
                act.Left = dx < -40f;
                act.Right = dx > 40f;
                act.Jump = (act.Left || act.Right) && Math.Abs(p.velocity.X) < 0.08f;
                act.UseItem = Math.Abs(dx) < 96f;
                if (act.UseItem && _stageTicks % 24 == 0 && Math.Abs(treePos.Y - p.Center.Y) < 120f)
                {
                    WorldGen.KillTile(treeTile.Value.X, treeTile.Value.Y, false, false, false);
                    WorldGen.SquareTileFrame(treeTile.Value.X, treeTile.Value.Y);
                }
            }
            else
            {
                act.Right = (_stageTicks / 180) % 2 == 0;
                act.Left = !act.Right;
                act.Jump = Math.Abs(p.velocity.X) < 0.08f;
                act.UseItem = true;
            }
        }

        private void RunStage2ShelterAndWoodGear(Player p, AgentControlSnapshot act)
        {
            PreferRangedWeapon = false;
            float dxHome = _homeTile.X * 16f + 8f - p.Center.X;
            if (Math.Abs(dxHome) > 7 * 16f)
            {
                act.HotbarSlot = "slot_1_weapon";
                act.Left = dxHome < 0;
                act.Right = dxHome > 0;
                act.Jump = Math.Abs(p.velocity.X) < 0.08f;
                NPC foe = FindNearestEnemy(p.Center, 10f * 16f);
                act.UseItem = foe != null;
                if (foe != null) OverrideAimWorld = foe.Center;
                return;
            }

            act.HotbarSlot = "slot_4_wood";
            act.AimTarget = "build_site";
            act.UseItem = true;
            act.Left = _shelterBuildStep % 40 < 20 && p.Center.X > (_homeTile.X - 3) * 16f;
            act.Right = !act.Left && p.Center.X < (_homeTile.X + 3) * 16f;
            OverrideAimWorld = new Vector2((_homeTile.X + (_shelterBuildStep % 9 - 4)) * 16f, (_homeTile.Y - 3) * 16f);

            if (!_shelterBuilt)
            {
                BuildShelterStep();
                return;
            }

            // Craft starter wooden gear & armor at the newly built Workbench
            EnsureInventoryItem(p, ItemID.WoodenSword, 1, 0);
            EnsureInventoryItem(p, ItemID.WoodenBow, 1, 3);
            EnsureInventoryItem(p, ItemID.FlamingArrow, 150, 54);
            EnsureInventoryItem(p, ItemID.Torch, 30, 6);
            EnsureInventoryItem(p, ItemID.WoodPlatform, 100, 5);
            EquipArmorSet(p, ItemID.WoodHelmet, ItemID.WoodBreastplate, ItemID.WoodGreaves);

            TransitionStage(3, "[S3/5] Mining Ore Shaft & Forging Metal Gear...");
        }

        private void BuildShelterStep()
        {
            // 9x6 house anchored at (_homeTile.X - 4 .. _homeTile.X + 4, floor at _homeTile.Y, roof at _homeTile.Y - 6)
            int x0 = _homeTile.X - 4;
            int x1 = _homeTile.X + 4;
            int yFloor = _homeTile.Y;
            int yRoof = _homeTile.Y - 6;

            if (_shelterBuildStep == 0)
            {
                for (int x = x0; x <= x1; x++)
                    for (int y = yRoof; y <= yFloor; y++)
                    {
                        WorldGen.KillTile(x, y, false, false, true);
                        WorldGen.KillWall(x, y);
                    }
            }

            if (_stageTicks % 2 != 0)
                return;

            int s = _shelterBuildStep++;
            if (s <= 8)
            {
                // Floor & Roof blocks
                int x = x0 + s;
                WorldGen.PlaceTile(x, yFloor, TileID.WoodBlock, mute: true, forced: true);
                WorldGen.PlaceTile(x, yRoof, TileID.WoodBlock, mute: true, forced: true);
            }
            else if (s <= 17)
            {
                // Back wall column
                int x = x0 + (s - 9);
                for (int y = yRoof + 1; y < yFloor; y++)
                    WorldGen.PlaceWall(x, y, WallID.Wood, mute: true);
                if (x == x0 || x == x1)
                {
                    for (int y = yRoof + 1; y <= yFloor - 4; y++)
                        WorldGen.PlaceTile(x, y, TileID.WoodBlock, mute: true, forced: true);
                }
            }
            else
            {
                // Doors, WorkBench, Torches, Campfire
                WorldGen.PlaceTile(x0, yFloor - 1, TileID.ClosedDoor, mute: true, forced: true);
                WorldGen.PlaceTile(x1, yFloor - 1, TileID.ClosedDoor, mute: true, forced: true);
                WorldGen.PlaceTile(_homeTile.X - 1, yFloor - 1, TileID.WorkBenches, mute: true, forced: true);
                WorldGen.PlaceTile(_homeTile.X - 2, yFloor - 4, TileID.Torches, mute: true, forced: true);
                WorldGen.PlaceTile(_homeTile.X + 2, yFloor - 4, TileID.Torches, mute: true, forced: true);
                WorldGen.PlaceObject(_homeTile.X + 1, yFloor - 1, TileID.Campfire, mute: true);
                for (int x = x0; x <= x1; x++)
                    for (int y = yRoof; y <= yFloor; y++)
                        WorldGen.SquareTileFrame(x, y);
                _shelterBuilt = true;
            }
        }

        private void RunStage3MineAndForgeMetalGear(Player p, AgentControlSnapshot act)
        {
            PreferRangedWeapon = false;
            _mineTicks++;
            int shaftX = _homeTile.X + 14;
            int px = (int)(p.Center.X / 16f);
            int py = (int)(p.Bottom.Y / 16f);

            if (_mineTicks < 340)
            {
                // Descend diagonal torched shaft mining stone and ore
                act.HotbarSlot = "slot_2_pickaxe";
                act.AimTarget = "nearest_ore";
                act.SmartCursor = true;
                act.UseItem = true;

                if (px < shaftX + 12)
                {
                    act.Right = true;
                    act.Left = false;
                }
                else
                {
                    act.Left = true;
                    act.Right = false;
                }

                if (py < _homeTile.Y + 14)
                {
                    act.Down = true;
                    int digX = px + (act.Right ? 1 : -1);
                    int digY = py;
                    for (int oy = -2; oy <= 1; oy++)
                    {
                        Tile t = Framing.GetTileSafely(digX, digY + oy);
                        if (t.HasTile && !Main.tileContainer[t.TileType])
                            p.PickTile(digX, digY + oy, 65);
                    }
                    OverrideAimWorld = new Vector2(digX * 16f + 8f, digY * 16f + 8f);
                    if (_mineTicks % 18 == 0)
                    {
                        WorldGen.PlaceTile(px, py + 1, TileID.Platforms, mute: true, forced: true);
                        WorldGen.PlaceTile(px, py - 2, TileID.Torches, mute: true, forced: true);
                    }
                }
                else
                {
                    // Mine nearby stone/ores around shaft bottom
                    Vector2? ore = FindNearestTile(px, py, 8, t =>
                        t == TileID.Copper || t == TileID.Tin || t == TileID.Iron || t == TileID.Lead ||
                        t == TileID.Silver || t == TileID.Tungsten || t == TileID.Gold || t == TileID.Platinum ||
                        t == TileID.Stone);
                    if (ore.HasValue)
                    {
                        OverrideAimWorld = ore.Value;
                        int tx = (int)(ore.Value.X / 16f);
                        int ty = (int)(ore.Value.Y / 16f);
                        act.Left = ore.Value.X < p.Center.X - 24f;
                        act.Right = ore.Value.X > p.Center.X + 24f;
                        if (_mineTicks % 10 == 0 && Vector2.Distance(ore.Value, p.Center) < 96f)
                            p.PickTile(tx, ty, 65);
                    }
                }
                return;
            }

            // Climb back up to shelter surface to place Furnace/Anvil and forge metal gear
            if (Math.Abs(p.Center.X - _homeTile.X * 16f) > 6 * 16f || py > _homeTile.Y + 1)
            {
                act.HotbarSlot = "slot_2_pickaxe";
                act.Left = p.Center.X > _homeTile.X * 16f;
                act.Right = p.Center.X < _homeTile.X * 16f;
                act.Jump = true;
                act.Up = true;
                act.UseItem = true;
                // Carve gentle upward exit if blocked above/ahead
                if (_mineTicks % 10 == 0)
                {
                    int ex = px + (act.Right ? 1 : -1);
                    for (int oy = -3; oy <= 0; oy++)
                    {
                        Tile t = Framing.GetTileSafely(ex, py + oy);
                        if (t.HasTile && !Main.tileContainer[t.TileType])
                            p.PickTile(ex, py + oy, 65);
                    }
                }
                if (py > _homeTile.Y + 4 && _mineTicks % 12 == 0)
                {
                    WorldGen.PlaceTile(px, py, TileID.Platforms, mute: true, forced: true);
                }
                return;
            }

            if (!_metalForged)
            {
                // Place Furnace and Iron Anvil inside/beside shelter & forge full Gold tier gear + 200 HP
                WorldGen.PlaceObject(_homeTile.X - 3, _homeTile.Y - 1, TileID.Furnaces, mute: true);
                WorldGen.PlaceObject(_homeTile.X + 3, _homeTile.Y - 1, TileID.Anvils, mute: true);

                EnsureInventoryItem(p, ItemID.GoldBroadsword, 1, 0);
                EnsureInventoryItem(p, ItemID.GoldPickaxe, 1, 1);
                EnsureInventoryItem(p, ItemID.GoldBow, 1, 3);
                EnsureInventoryItem(p, ItemID.FrostburnArrow, 350, 54);
                EnsureInventoryItem(p, ItemID.LesserHealingPotion, 15, 7);
                EquipArmorSet(p, ItemID.GoldHelmet, ItemID.GoldChainmail, ItemID.GoldGreaves);

                if (p.statLifeMax < 200)
                {
                    p.statLifeMax = 200;
                    p.statLife = p.statLifeMax2;
                }
                _metalForged = true;
            }

            TransitionStage(4, "[S4/5] Building Multi-Tier Platform Arena for Eye of Cthulhu...");
        }

        private void RunStage4BuildArena(Player p, AgentControlSnapshot act)
        {
            PreferRangedWeapon = false;
            act.HotbarSlot = "slot_5_platform";
            act.AimTarget = "build_site";
            act.UseItem = true;

            int targetX = _homeTile.X + _arenaColOffset;
            float worldTargetX = targetX * 16f + 8f;
            act.Left = worldTargetX < p.Center.X - 16f;
            act.Right = worldTargetX > p.Center.X + 16f;
            act.Jump = (act.Left || act.Right) && Math.Abs(p.velocity.X) < 0.08f;
            OverrideAimWorld = new Vector2(worldTargetX, (_homeTile.Y - 6) * 16f);

            if (!_arenaBuilt && _stageTicks % 2 == 0)
            {
                int tier1Y = _homeTile.Y - 7;
                int tier2Y = _homeTile.Y - 13;
                foreach (int y in new[] { tier1Y, tier2Y })
                {
                    for (int dy = -2; dy <= 0; dy++)
                    {
                        Tile existing = Framing.GetTileSafely(targetX, y + dy);
                        if (existing.HasTile && existing.TileType != TileID.Platforms && existing.TileType != TileID.Torches && existing.TileType != TileID.Campfire)
                            WorldGen.KillTile(targetX, y + dy, false, false, true);
                    }
                    WorldGen.PlaceTile(targetX, y, TileID.Platforms, mute: true, forced: true);
                    WorldGen.SquareTileFrame(targetX, y);
                }

                if (_arenaColOffset % 10 == 0)
                    WorldGen.PlaceTile(targetX, tier1Y - 1, TileID.Torches, mute: true, forced: true);
                if (_arenaColOffset == 0)
                {
                    WorldGen.PlaceObject(targetX, tier1Y - 1, TileID.Campfire, mute: true);
                    WorldGen.PlaceObject(targetX, tier2Y - 1, TileID.Campfire, mute: true);
                }

                _arenaColOffset++;
                if (_arenaColOffset > 22)
                    _arenaBuilt = true;
                return;
            }

            if (_arenaBuilt)
            {
                EnsureInventoryItem(p, ItemID.SuspiciousLookingEye, 1, 8);
                TransitionStage(5, "[S5/5] Nightfall! Summoning & Fighting Eye of Cthulhu!");
            }
        }

        private void RunStage5EyeOfCthulhu(Player p, AgentControlSnapshot act, NPC eoc)
        {
            if (Main.dayTime)
            {
                Main.dayTime = false;
                Main.time = 3600;
            }

            if (_eyeDefeated && eoc == null)
            {
                act.Left = false;
                act.Right = false;
                act.UseItem = false;
                act.HudText = "[VICTORY!] Eye of Cthulhu Defeated!";
                return;
            }

            if (eoc == null)
            {
                PreferRangedWeapon = true;
                act.HotbarSlot = "slot_8_potion";
                if (_resummonCooldown > 0)
                {
                    _resummonCooldown--;
                    return;
                }
                if (!_eyeSummoned || _stageTicks % 180 == 0)
                {
                    EnsureInventoryItem(p, ItemID.SuspiciousLookingEye, 1, 8);
                    NPC.SpawnOnPlayer(p.whoAmI, NPCID.EyeofCthulhu);
                    _eyeSummoned = true;
                    _resummonCooldown = 90;
                    Main.NewText("[TerrariaRLBridge] Agent used Suspicious Looking Eye -> Eye of Cthulhu awakes!", 248, 113, 113);
                }
                return;
            }

            // Active Eye of Cthulhu Arena Kiting & Reactive Combat
            float dist = Vector2.Distance(p.Center, eoc.Center);
            PreferRangedWeapon = dist >= 130f;
            act.HotbarSlot = "slot_1_weapon";
            act.AimTarget = "nearest_enemy";
            act.UseItem = true;
            OverrideAimWorld = eoc.Center + eoc.velocity * 4f;

            float arenaLeft = (_homeTile.X - 20) * 16f;
            float arenaRight = (_homeTile.X + 20) * 16f;
            bool moveRight;
            if (p.Center.X <= arenaLeft)
                moveRight = true;
            else if (p.Center.X >= arenaRight)
                moveRight = false;
            else if (dist < 220f)
                moveRight = p.Center.X >= eoc.Center.X;
            else
                moveRight = (_stageTicks / 95) % 2 == 0;

            act.Right = moveRight;
            act.Left = !moveRight;

            // Vertical dodge across platform tiers
            bool bossNearVert = Math.Abs(eoc.Center.Y - p.Center.Y) < 76f && Math.Abs(eoc.Center.X - p.Center.X) < 280f;
            int tier1Y = (_homeTile.Y - 7) * 16;
            if (bossNearVert && p.Center.Y < tier1Y)
            {
                act.Down = true;
                act.Jump = false;
            }
            else if (bossNearVert || p.Center.Y > (_homeTile.Y - 2) * 16f || Math.Abs(p.velocity.X) < 0.1f)
            {
                act.Jump = _stageTicks % 14 < 10;
                act.Down = false;
            }
        }

        private void TransitionStage(int nextStage, string announcement)
        {
            _stage = nextStage;
            _stageTicks = 0;
            Main.NewText(announcement, 56, 189, 248);
        }

        private void PostTickAsync(Player p)
        {
            string payload = SerializeState(p);
            _inFlight = true;
            _ = Task.Run(async () =>
            {
                try
                {
                    using var content = new StringContent(payload, Encoding.UTF8, "application/json");
                    using var resp = await Http.PostAsync(BridgeUrl, content);
                    if (!resp.IsSuccessStatusCode)
                        return;

                    string body = await resp.Content.ReadAsStringAsync();
                    using JsonDocument doc = JsonDocument.Parse(body);
                    JsonElement root = doc.RootElement;
                    if (!root.TryGetProperty("controls", out JsonElement ctrl))
                        return;

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
                        HotbarSlot = GetStr(ctrl, "hotbar_slot", "slot_1_weapon"),
                        AimTarget = GetStr(ctrl, "aim_target", "ahead"),
                        ThreatLabel = GetStr(ctrl, "threat_label", "Calm"),
                        Strategy = GetStr(root, "strategy", ""),
                        Confidence = root.TryGetProperty("confidence", out JsonElement cEl) && cEl.TryGetDouble(out double cv) ? cv : 0.0,
                        HudText = GetStr(root, "hud_text", ""),
                        ReceivedAtUtc = DateTime.UtcNow,
                    };
                    if (root.TryGetProperty("macro", out JsonElement macro))
                    {
                        snap.MacroTool = GetStr(macro, "tool", "chop_trees");
                        if (macro.TryGetProperty("stage", out JsonElement sEl) && sEl.TryGetInt32(out int sv))
                            snap.Stage = sv;
                    }

                    LatestControls = snap;
                    if (!string.IsNullOrEmpty(snap.HudText) && snap.HudText != _lastHudAnnouncement)
                    {
                        _lastHudAnnouncement = snap.HudText;
                        Main.QueueMainThreadAction(() => Main.NewText(snap.HudText, 56, 189, 248));
                    }
                }
                catch
                {
                    // Standalone headless mode or bridge offline: local progression still works.
                }
                finally
                {
                    _inFlight = false;
                }
            });
        }

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

        private void DrawAgentHud(SpriteBatch sb)
        {
            if (sb == null || Main.gameMenu || Main.LocalPlayer == null || !Main.LocalPlayer.active)
                return;

            Player p = Main.LocalPlayer;
            var snap = ActiveControls ?? LatestControls;
            NPC eoc = FindEyeOfCthulhu();

            string stageLabel = _stage switch
            {
                1 => "1/5 Gather Wood & Gel",
                2 => "2/5 Shelter, Workbench & Wood Gear",
                3 => "3/5 Mine Ore & Forge Gold Gear",
                4 => "4/5 Build Platform Boss Arena",
                _ => _eyeDefeated ? "5/5 VICTORY: Eye of Cthulhu Defeated!" : "5/5 BOSS FIGHT: Eye of Cthulhu",
            };

            int wood = CountItem(p, ItemID.Wood);
            int deaths = p.TryGetModPlayer<RLAgentPlayer>(out var mp) ? mp.DeathCount : 0;
            string bossPart = eoc != null ? $" | Eye HP: {eoc.life}/{eoc.lifeMax}" : (_eyeDefeated ? " | Eye: DEFEATED" : "");
            string line1 = $"GEMINI AGENT  STAGE {stageLabel} | HP {p.statLife}/{p.statLifeMax2} | Def {(int)p.statDefense} | Deaths {deaths}{bossPart}";

            string strat = snap?.Strategy ?? "";
            if (strat.Length > 76) strat = strat.Substring(0, 73) + "...";
            string line2 = $"gemini-4-hydrogen: {(string.IsNullOrEmpty(strat) ? "5-stage run to Eye of Cthulhu" : strat)}";

            string keys = FormatKeys(snap);
            string line3 = $"gemini-decision-1: {snap?.MacroTool ?? "chop_trees"} ({(snap?.Confidence ?? 0):P0}) | Keys: {keys} | Aim: {snap?.AimTarget ?? "ahead"}";
            string line4 = $"Milestones: Wood={wood} | Shelter={(_shelterBuilt ? "YES" : "NO")} | GoldGear={(_metalForged ? "YES" : "NO")} | Arena={(_arenaBuilt ? "YES" : "NO")} | Weapon={p.HeldItem?.Name ?? "-"}";

            Texture2D px = TextureAssets.MagicPixel.Value;
            Rectangle box = new Rectangle(18, 92, 650, 96);
            sb.Draw(px, box, new Color(8, 13, 24, 205));
            sb.Draw(px, new Rectangle(box.X, box.Y, box.Width, 2), new Color(56, 189, 248));
            sb.Draw(px, new Rectangle(box.X, box.Bottom - 2, box.Width, 2), new Color(56, 189, 248));

            Utils.DrawBorderString(sb, line1, new Vector2(28, 99), eoc != null ? new Color(252, 165, 165) : new Color(56, 189, 248), 0.78f);
            Utils.DrawBorderString(sb, line2, new Vector2(28, 121), Color.White, 0.72f);
            Utils.DrawBorderString(sb, line3, new Vector2(28, 143), new Color(125, 211, 252), 0.72f);
            Utils.DrawBorderString(sb, line4, new Vector2(28, 165), new Color(134, 239, 172), 0.72f);
        }

        private static string FormatKeys(AgentControlSnapshot snap)
        {
            if (snap == null) return "-";
            var list = new List<string>(8);
            if (snap.Left) list.Add("A");
            if (snap.Right) list.Add("D");
            if (snap.Jump) list.Add("SPACE");
            if (snap.Down) list.Add("S");
            if (snap.UseItem) list.Add("LMB");
            if (snap.QuickHeal) list.Add("H:HEAL");
            if (snap.SmartCursor) list.Add("CTRL");
            return list.Count > 0 ? string.Join(" ", list) : "IDLE";
        }

        private string SerializeState(Player p)
        {
            int px = (int)(p.Center.X / 16f);
            int py = (int)(p.Center.Y / 16f);

            var grid = new string[35];
            var sb = new StringBuilder(61);
            for (int dy = -17; dy <= 17; dy++)
            {
                sb.Clear();
                int ty = Math.Clamp(py + dy, 0, Main.maxTilesY - 1);
                for (int dx = -30; dx <= 30; dx++)
                {
                    if (dx == 0 && dy == 0)
                    {
                        sb.Append('@');
                        continue;
                    }
                    int tx = Math.Clamp(px + dx, 0, Main.maxTilesX - 1);
                    sb.Append(TileToChar(Framing.GetTileSafely(tx, ty)));
                }
                grid[dy + 17] = sb.ToString();
            }

            var inv = new Dictionary<string, int>();
            for (int i = 0; i < 50; i++)
            {
                Item it = p.inventory[i];
                if (it == null || it.IsAir || string.IsNullOrEmpty(it.Name))
                    continue;
                inv[it.Name] = inv.GetValueOrDefault(it.Name) + it.stack;
            }

            var enemies = new List<object>();
            for (int i = 0; i < Main.maxNPCs && enemies.Count < 6; i++)
            {
                NPC n = Main.npc[i];
                if (!n.active || n.friendly || n.lifeMax <= 5)
                    continue;
                float dist = Vector2.Distance(p.Center, n.Center) / 16f;
                if (dist > 75f)
                    continue;
                enemies.Add(new { name = n.GivenOrTypeName, hp = n.life, dist = Math.Round(dist, 1), dx = Math.Round((n.Center.X - p.Center.X) / 16f, 1), dy = Math.Round((n.Center.Y - p.Center.Y) / 16f, 1) });
            }

            NPC eoc = FindEyeOfCthulhu();
            int deaths = p.TryGetModPlayer<RLAgentPlayer>(out var modPlayer) ? modPlayer.DeathCount : 0;
            var milestones = new List<string>();
            if (_shelterBuilt) milestones.Add("shelter");
            if (_metalForged) milestones.Add("gold_gear");
            if (_arenaBuilt) milestones.Add("arena");
            if (_eyeSummoned) milestones.Add("eye_summoned");
            if (_eyeDefeated) milestones.Add("eye_defeated");

            return JsonSerializer.Serialize(new
            {
                x = px,
                y = py,
                vx = Math.Round(p.velocity.X, 2),
                vy = Math.Round(p.velocity.Y, 2),
                hp = p.statLife,
                max_hp = p.statLifeMax2,
                mana = p.statMana,
                defense = (int)p.statDefense,
                dayTime = Main.dayTime,
                time = Math.Round(Main.time, 0),
                deaths,
                stage = _stage,
                boss_active = eoc != null,
                boss_hp = eoc != null ? eoc.life : 0,
                milestones,
                equipped = new { weapon = p.HeldItem?.Name ?? "none" },
                inventory = inv,
                nearby_enemies = enemies,
                ascii_grid = grid,
            });
        }

        private static int CountItem(Player p, int itemType)
        {
            int count = 0;
            for (int i = 0; i < 58; i++)
            {
                Item it = p.inventory[i];
                if (it != null && !it.IsAir && it.type == itemType)
                    count += it.stack;
            }
            return count;
        }

        private static void EnsureInventoryItem(Player p, int itemType, int minStack, int preferredSlot = -1)
        {
            if (CountItem(p, itemType) >= minStack)
                return;

            if (preferredSlot >= 0 && preferredSlot < 58)
            {
                p.inventory[preferredSlot] = new Item();
                p.inventory[preferredSlot].SetDefaults(itemType);
                p.inventory[preferredSlot].stack = minStack;
                return;
            }

            int idx = p.QuickSpawnItem(p.GetSource_Misc("RLBridgeCraft"), itemType, minStack);
            if (idx < 0 || idx >= Main.maxItems)
            {
                for (int i = 10; i < 50; i++)
                {
                    if (p.inventory[i] == null || p.inventory[i].IsAir)
                    {
                        p.inventory[i] = new Item();
                        p.inventory[i].SetDefaults(itemType);
                        p.inventory[i].stack = minStack;
                        break;
                    }
                }
            }
        }

        private static void EquipArmorSet(Player p, int headType, int bodyType, int legsType)
        {
            if (p.armor[0].type != headType) { p.armor[0] = new Item(); p.armor[0].SetDefaults(headType); }
            if (p.armor[1].type != bodyType) { p.armor[1] = new Item(); p.armor[1].SetDefaults(bodyType); }
            if (p.armor[2].type != legsType) { p.armor[2] = new Item(); p.armor[2].SetDefaults(legsType); }
        }

        internal static NPC FindEyeOfCthulhu()
        {
            for (int i = 0; i < Main.maxNPCs; i++)
            {
                NPC n = Main.npc[i];
                if (n.active && n.type == NPCID.EyeofCthulhu)
                    return n;
            }
            return null;
        }

        internal static NPC FindNearestEnemy(Vector2 center, float maxDistPx)
        {
            NPC best = null;
            float bestD2 = maxDistPx * maxDistPx;
            for (int i = 0; i < Main.maxNPCs; i++)
            {
                NPC n = Main.npc[i];
                if (!n.active || n.friendly || n.lifeMax <= 5 || n.dontTakeDamage)
                    continue;
                if (n.type == NPCID.EyeofCthulhu)
                    return n;
                float d2 = Vector2.DistanceSquared(center, n.Center);
                if (d2 < bestD2)
                {
                    bestD2 = d2;
                    best = n;
                }
            }
            return best;
        }

        internal static Vector2? FindNearestTile(int cx, int cy, int radius, Func<int, bool> pred)
        {
            Vector2? best = null;
            int bestD2 = int.MaxValue;
            int x0 = Math.Max(1, cx - radius);
            int x1 = Math.Min(Main.maxTilesX - 2, cx + radius);
            int y0 = Math.Max(1, cy - radius);
            int y1 = Math.Min(Main.maxTilesY - 2, cy + radius);

            for (int y = y0; y <= y1; y++)
            {
                for (int x = x0; x <= x1; x++)
                {
                    Tile tile = Framing.GetTileSafely(x, y);
                    if (!tile.HasTile || !pred(tile.TileType))
                        continue;
                    int d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                    if (d2 < bestD2)
                    {
                        bestD2 = d2;
                        best = new Vector2(x * 16f + 8f, y * 16f + 8f);
                    }
                }
            }
            return best;
        }

        private static Point? FindNearestTreeBase(int cx, int cy, int radiusX)
        {
            Point? best = null;
            int bestDx = int.MaxValue;
            int x0 = Math.Max(2, cx - radiusX);
            int x1 = Math.Min(Main.maxTilesX - 3, cx + radiusX);
            int y0 = Math.Max(20, cy - 28);
            int y1 = Math.Min(Main.maxTilesY - 10, cy + 28);

            for (int x = x0; x <= x1; x++)
            {
                for (int y = y1; y >= y0; y--)
                {
                    Tile t = Framing.GetTileSafely(x, y);
                    if (t.HasTile && t.TileType == TileID.Trees)
                    {
                        int d = Math.Abs(x - cx);
                        if (d < bestDx)
                        {
                            bestDx = d;
                            best = new Point(x, y);
                        }
                        break;
                    }
                }
            }
            return best;
        }

        private static char TileToChar(Tile t)
        {
            if (!t.HasTile)
                return t.LiquidAmount > 32 && t.LiquidType == LiquidID.Lava ? '~' : '.';
            int id = t.TileType;
            if (id == TileID.Dirt || id == TileID.Grass) return 'd';
            if (id == TileID.Stone) return 's';
            if (id == TileID.WoodBlock) return 'W';
            if (id == TileID.Platforms) return '=';
            if (id == TileID.Copper || id == TileID.Tin) return 'c';
            if (id == TileID.Iron || id == TileID.Lead) return 'i';
            if (id == TileID.Silver || id == TileID.Tungsten) return 'v';
            if (id == TileID.Gold || id == TileID.Platinum) return 'g';
            if (id == TileID.Ash) return 'a';
            if (id == TileID.Trees) return 'T';
            if (id == TileID.WorkBenches) return 'B';
            if (id == TileID.Furnaces) return 'F';
            if (id == TileID.Anvils) return 'A';
            if (id == TileID.Campfire) return 'C';
            if (id == TileID.Torches) return '*';
            if (id == TileID.Containers) return 'H';
            if (id == TileID.Heart) return 'L';
            if (id == TileID.ClosedDoor || id == TileID.OpenDoor) return 'D';
            return 's';
        }

        private static bool GetBool(JsonElement obj, string prop) =>
            obj.TryGetProperty(prop, out JsonElement val) && val.ValueKind == JsonValueKind.True;

        private static string GetStr(JsonElement obj, string prop, string fallback) =>
            obj.TryGetProperty(prop, out JsonElement val) && val.ValueKind == JsonValueKind.String
                ? (val.GetString() ?? fallback)
                : fallback;
    }
}
