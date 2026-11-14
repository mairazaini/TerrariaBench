using System;
using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Terraria;
using Terraria.ID;
using Terraria.ModLoader;
using Terraria.ObjectData;
using Terraria.UI;

namespace TerrariaRLBridge
{
    /// <summary>
    /// HONEST SKILLS. A skill is a short closed-loop program that plays the way a person does: it selects a hotbar
    /// item, points the mouse, holds the mouse button and presses movement keys, 60 times a second. A skill never
    /// edits tiles, never moves items or the player, never deals damage. Blocks break at the speed of the held
    /// pickaxe (or not at all if its power is too low), trees take real axe swings, drops are picked up by walking
    /// over them, arrows are real projectiles. The only non-input calls are the same UI clicks the bridge already
    /// had (craft from the recipe list, equip from the inventory, Loot All on an open chest).
    ///
    /// The Python side starts a skill by sending controls.skill = {id, name, args}. Progress is reported back in
    /// hud.skill = {id, name, status: idle|running|success|failed, reason, ticks, detail}.
    /// Coordinates in args are tiles relative to the world spawn point (x right, y down).
    /// </summary>
    internal static class SkillRunner
    {
        internal static int Id;
        internal static string Name = "";
        internal static string Status = "idle";
        internal static string Reason = "";
        internal static string Detail = "";
        internal static int Ticks;

        private static Dictionary<string, string> _args = new Dictionary<string, string>();
        private static int _timeout;
        private static int _phase;
        private static int _phaseTicks;
        private static int _startX, _startY;
        private static Point _target;
        private static bool _hasTarget;
        private static int _counter;
        private static int _baseline;

        // input edge state
        private static bool _usePressed, _useReq, _jumpHeld, _jumpReq;
        // walker state
        private static float _lastX;
        private static int _stuck;
        // mining / placing state
        private static int _mineX = -1, _mineY = -1, _mineTicks, _approachTicks;
        private static int _placeTicks;
        // combat state
        private static int _runDir, _edgeHold, _wetTicks;

        internal static bool Active => Status == "running";

        internal static object Hud()
        {
            return new { id = Id, name = Name, status = Status, reason = Reason, ticks = Ticks, detail = Detail };
        }

        internal static void Reset()
        {
            Id = 0;
            Name = "";
            Status = "idle";
            Reason = "";
            Detail = "";
            Ticks = 0;
        }

        /// <summary>Starts or cancels a skill when the latest bridge reply asks for a different one.</summary>
        internal static void Sync(AgentControlSnapshot snap, Player p)
        {
            if (snap == null || snap.SkillId == 0 || snap.SkillId == Id)
                return;
            Id = snap.SkillId;
            Name = (snap.SkillName ?? "").Trim().ToLowerInvariant();
            _args = snap.SkillArgs ?? new Dictionary<string, string>();
            Ticks = 0;
            Reason = "";
            Detail = "";
            _phase = 0;
            _phaseTicks = 0;
            _hasTarget = false;
            _counter = 0;
            _baseline = 0;
            _stuck = 0;
            _mineX = _mineY = -1;
            _mineTicks = _approachTicks = _placeTicks = 0;
            _runDir = 0;
            _edgeHold = 0;
            _wetTicks = 0;
            _usePressed = _jumpHeld = false;
            _startX = Cx(p);
            _startY = Cy(p);
            if (Name == "" || Name == "cancel" || Name == "none")
            {
                Status = "idle";
                Reason = "cancelled";
                return;
            }
            Status = "running";
            _timeout = ArgI("timeout_s", DefaultTimeoutSeconds(Name)) * 60;
            if (Name != "craft" && Name != "equip" && Name != "loot_chest" && Main.playerInventory)
            {
                Main.playerInventory = false;
                p.chest = -1;
            }
        }

        private static int DefaultTimeoutSeconds(string name)
        {
            switch (name)
            {
                case "goto": return 120;
                case "explore": return 150;
                case "chop": return 300;
                case "mine": return 300;
                case "dig_down": return 300;
                case "surface": return 420;
                case "craft": return 90;
                case "place": return 40;
                case "equip": return 5;
                case "use": return 12;
                case "build_arena": return 420;
                case "fight": return 660;
                case "hunt": return 600;
                case "loot_chest": return 90;
                case "wait": return 1000;
                default: return 60;
            }
        }

        /// <summary>Runs one 60 Hz step of the active skill. Called from ModPlayer.SetControls.</summary>
        internal static void Tick(Player p)
        {
            if (!Active)
                return;
            ClearControls(p);
            Ticks++;
            _phaseTicks++;
            if (p.dead)
            {
                Fail("player died");
                return;
            }
            if (Ticks > _timeout)
            {
                Fail("timeout after " + (_timeout / 60) + "s" + (Detail == "" ? "" : " (" + Detail + ")"));
                return;
            }
            try
            {
                switch (Name)
                {
                    case "goto": Goto(p); break;
                    case "explore": Explore(p); break;
                    case "chop": Chop(p); break;
                    case "mine": Mine(p); break;
                    case "dig_down": DigDown(p); break;
                    case "surface": Surface(p); break;
                    case "craft": Craft(p); break;
                    case "place": Place(p); break;
                    case "equip": Equip(p); break;
                    case "use": UseItemSkill(p); break;
                    case "build_arena": BuildArena(p); break;
                    case "fight": Fight(p); break;
                    case "hunt": Hunt(p); break;
                    case "loot_chest": LootChest(p); break;
                    case "wait": Wait(p); break;
                    default: Fail("unknown skill '" + Name + "'"); break;
                }
            }
            catch (Exception e)
            {
                Fail("skill error: " + e.GetType().Name + ": " + e.Message);
            }
            if (!_useReq) _usePressed = false;
            if (!_jumpReq) _jumpHeld = false;
        }

        private static void Done(string reason)
        {
            Status = "success";
            Reason = reason;
            ModContent.GetInstance<MeteredBridgeSystem>().AddEvent("skill " + Name + " succeeded: " + reason);
        }

        private static void Fail(string reason)
        {
            Status = "failed";
            Reason = reason;
            ModContent.GetInstance<MeteredBridgeSystem>().AddEvent("skill " + Name + " failed: " + reason);
        }

        private static void SetPhase(int phase)
        {
            _phase = phase;
            _phaseTicks = 0;
        }

        // ------------------------------------------------------------------ args

        private static string ArgS(string key, string fallback)
        {
            return _args.TryGetValue(key, out string v) && !string.IsNullOrWhiteSpace(v) ? v.Trim() : fallback;
        }

        private static int ArgI(string key, int fallback)
        {
            return _args.TryGetValue(key, out string v) && double.TryParse(v, System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out double d) ? (int)Math.Round(d) : fallback;
        }

        private static bool HasArg(string key) => _args.ContainsKey(key);

        // ------------------------------------------------------------------ input primitives

        private static void ClearControls(Player p)
        {
            p.controlLeft = p.controlRight = p.controlUp = p.controlDown = false;
            p.controlJump = p.controlUseItem = p.controlUseTile = false;
            p.controlHook = p.controlQuickHeal = p.controlThrow = false;
            _useReq = _jumpReq = false;
        }

        /// <summary>Points the mouse at a world position with Smart Cursor off, so the click lands exactly there.</summary>
        private static void Aim(Vector2 world)
        {
            Main.SmartCursorWanted_Mouse = false;
            Main.mouseX = (int)Math.Clamp(world.X - Main.screenPosition.X, 4, Main.screenWidth - 4);
            Main.mouseY = (int)Math.Clamp(world.Y - Main.screenPosition.Y, 4, Main.screenHeight - 4);
            Terraria.GameInput.PlayerInput.MouseX = Main.mouseX;
            Terraria.GameInput.PlayerInput.MouseY = Main.mouseY;
        }

        private static void AimTile(int x, int y) => Aim(new Vector2(x * 16f + 8f, y * 16f + 8f));

        /// <summary>Holds the left mouse button, re-clicking between swings for items that do not auto-reuse.</summary>
        private static void Use(Player p)
        {
            _useReq = true;
            if (_usePressed && p.itemAnimation == 0 && p.itemTime == 0)
            {
                p.controlUseItem = false;
                p.releaseUseItem = true;
                _usePressed = false;
            }
            else
            {
                p.controlUseItem = true;
                _usePressed = true;
            }
        }

        /// <summary>Holds Space, re-tapping on landing like a player.</summary>
        private static void Jump(Player p)
        {
            _jumpReq = true;
            if (_jumpHeld && p.velocity.Y == 0f)
            {
                p.controlJump = false;
                p.releaseJump = true;
                _jumpHeld = false;
            }
            else
            {
                p.controlJump = true;
                _jumpHeld = true;
            }
        }

        private static int Cx(Player p) => (int)(p.Center.X / 16f);
        private static int Cy(Player p) => (int)(p.Center.Y / 16f);
        private static int FeetRow(Player p) => (int)((p.Bottom.Y - 1f) / 16f);
        private static int TopRow(Player p) => (int)(p.position.Y / 16f);
        private static int LeftCol(Player p) => (int)(p.position.X / 16f);
        private static int RightCol(Player p) => (int)((p.position.X + p.width - 1f) / 16f);
        private static int AbsX(int rel) => Main.spawnTileX + rel;
        private static int AbsY(int rel) => Main.spawnTileY + rel;

        private static bool Solid(int x, int y)
        {
            if (!WorldGen.InWorld(x, y, 2))
                return true;
            Tile t = Framing.GetTileSafely(x, y);
            return t.HasTile && !t.IsActuated && Main.tileSolid[t.TileType] && !Main.tileSolidTop[t.TileType];
        }

        private static bool HasTile(int x, int y)
        {
            return WorldGen.InWorld(x, y, 2) && Framing.GetTileSafely(x, y).HasTile;
        }

        private static bool Lava(int x, int y)
        {
            if (!WorldGen.InWorld(x, y, 2))
                return false;
            Tile t = Framing.GetTileSafely(x, y);
            return t.LiquidAmount > 0 && t.LiquidType == LiquidID.Lava;
        }

        /// <summary>The game's own tool reach test, one tile tighter sideways and upward to stay clear of the edge.</summary>
        private static bool InReach(Player p, int x, int y, int boost)
        {
            float left = p.position.X / 16f - Player.tileRangeX - boost + 1f;
            float right = (p.position.X + p.width) / 16f + Player.tileRangeX + boost - 2f;
            float top = p.position.Y / 16f - Player.tileRangeY - boost + 1f;
            float bottom = (p.position.Y + p.height) / 16f + Player.tileRangeY + boost - 2f;
            return x >= left && x <= right && y >= top && y <= bottom;
        }

        private static int Best(Player p, Func<Item, int> score)
        {
            int best = -1, bestScore = 0;
            for (int i = 0; i < 50; i++)
            {
                Item it = p.inventory[i];
                if (it == null || it.IsAir || it.stack <= 0)
                    continue;
                int s = score(it);
                if (s > bestScore)
                {
                    bestScore = s;
                    best = i;
                }
            }
            return best;
        }

        private static int SlotByName(Player p, string name)
        {
            return Best(p, it => string.Equals(it.Name, name, StringComparison.OrdinalIgnoreCase) ? 1 : 0);
        }

        private static int Count(Player p, Func<Item, bool> match)
        {
            int n = 0;
            for (int i = 0; i < 58; i++)
            {
                Item it = p.inventory[i];
                if (it != null && !it.IsAir && it.stack > 0 && match(it))
                    n += it.stack;
            }
            return n;
        }

        private static int CountName(Player p, string name)
        {
            return Count(p, it => string.Equals(it.Name, name, StringComparison.OrdinalIgnoreCase));
        }

        /// <summary>
        /// Makes the given inventory slot the held item (hotbar key, or dragging it into the last hotbar slot).
        /// Only switches between swings. Returns true once the item is in hand.
        /// </summary>
        private static bool Hold(Player p, int slot)
        {
            if (slot < 0)
                return false;
            if (slot < 10 && p.selectedItem == slot)
                return true;
            if (p.itemAnimation > 0 || p.itemTime > 0)
                return false;
            if (slot < 10)
            {
                p.selectedItem = slot;
                return false;
            }
            Utils.Swap(ref p.inventory[slot], ref p.inventory[9]);
            p.selectedItem = 9;
            return false;
        }

        private static int BlockScore(Item it)
        {
            if (it.createTile < 0 || it.createTile >= Main.tileSolid.Length)
                return 0;
            if (!Main.tileSolid[it.createTile] || Main.tileSolidTop[it.createTile])
                return 0;
            if (TileID.Sets.Ore[it.createTile] || TileID.Sets.Falling[it.createTile])
                return 0;
            if (it.type == ItemID.DirtBlock) return 100;
            if (it.type == ItemID.MudBlock) return 90;
            if (it.type == ItemID.StoneBlock) return 40;
            if (it.type == ItemID.Wood) return 10;
            return 60;
        }

        // ------------------------------------------------------------------ world primitives

        /// <summary>
        /// Swings the right tool at one tile. 1 = tile is gone, 0 = still working, -1 = gave up (and the skill failed).
        /// </summary>
        private static int MineStep(Player p, int x, int y)
        {
            Tile t = Framing.GetTileSafely(x, y);
            if (!t.HasTile)
                return 1;
            if (_mineX != x || _mineY != y)
            {
                _mineX = x;
                _mineY = y;
                _mineTicks = 0;
            }
            bool tree = MeteredBridgeSystem.IsTreeTile(t);
            int slot = tree ? Best(p, it => it.axe) : Best(p, it => it.pick);
            if (slot < 0)
            {
                Fail(tree ? "no axe in the inventory" : "no pickaxe in the inventory");
                return -1;
            }
            _mineTicks++;
            if (_mineTicks > 600)
            {
                Fail($"tile '{TileID.Search.GetName(t.TileType)}' did not break in 10s (tool power too low or out of reach)");
                return -1;
            }
            AimTile(x, y);
            if (Hold(p, slot))
                Use(p);
            return 0;
        }

        /// <summary>Walks/digs until the tile is in tool reach, then mines it. Same return codes as MineStep.</summary>
        private static int ApproachMine(Player p, int x, int y)
        {
            if (!HasTile(x, y))
                return 1;
            if (InReach(p, x, y, -1))
            {
                _approachTicks = 0;
                return MineStep(p, x, y);
            }
            if (++_approachTicks > 2400)
            {
                Fail("could not get within reach of the target tile");
                return -1;
            }
            StepToward(p, x, y, 0, 0, true, 40);
            return 0;
        }

        /// <summary>Uses the held placeable on an empty tile. 1 = placed, 0 = working, -1 = this spot did not work.</summary>
        private static int PlaceStep(Player p, int x, int y, int slot)
        {
            if (HasTile(x, y))
            {
                _placeTicks = 0;
                return 1;
            }
            if (slot < 0)
                return -1;
            if (++_placeTicks > 150)
            {
                _placeTicks = 0;
                return -1;
            }
            AimTile(x, y);
            if (Hold(p, slot))
                Use(p);
            return 0;
        }

        /// <summary>
        /// One step of honest travel toward a tile: walk, jump over what blocks the way, and when stuck for
        /// digDelay ticks mine through it. Below: dig the floor. Above: pillar up on placed blocks.
        /// Returns true on arrival.
        /// </summary>
        private static bool StepToward(Player p, int tx, int ty, int tolX, int tolY, bool dig, int digDelay)
        {
            int px = Cx(p), py = Cy(p);
            int dx = tx - px, dy = ty - py;
            if (Math.Abs(dx) <= tolX && Math.Abs(dy) <= tolY)
                return true;
            int top = TopRow(p), feet = FeetRow(p);

            if (Math.Abs(dx) > tolX)
            {
                int dir = Math.Sign(dx);
                if (dir > 0) p.controlRight = true; else p.controlLeft = true;
                int col = dir > 0 ? (int)((p.position.X + p.width + 3f) / 16f) : (int)((p.position.X - 3f) / 16f);
                bool blocked = false;
                for (int r = top; r <= feet; r++)
                    if (Solid(col, r)) blocked = true;

                if (Math.Abs(p.position.X - _lastX) < 0.3f) _stuck++; else _stuck = 0;
                _lastX = p.position.X;

                if (dig && _stuck > digDelay)
                {
                    for (int r = top - 1; r <= feet; r++)
                    {
                        if (Solid(col, r))
                        {
                            MineStep(p, col, r);
                            return false;
                        }
                    }
                }
                if (blocked || _stuck > 8)
                    Jump(p);
                return false;
            }

            int lc = LeftCol(p), rc = RightCol(p);
            if (dy > 0)
            {
                p.controlDown = true; // drop through platforms
                if (!dig)
                    return false;
                for (int c = lc; c <= rc; c++)
                {
                    if (Solid(c, feet + 1))
                    {
                        MineStep(p, c, feet + 1);
                        return false;
                    }
                }
                return false; // falling
            }
            PillarStep(p);
            return false;
        }

        /// <summary>Climbs one tile: clear the ceiling, jump, and put a block on the floor under the feet mid-air.</summary>
        private static void PillarStep(Player p)
        {
            int top = TopRow(p), feet = FeetRow(p), px = Cx(p);
            for (int c = LeftCol(p); c <= RightCol(p); c++)
            {
                for (int r = top - 1; r >= top - 2; r--)
                {
                    if (Solid(c, r))
                    {
                        MineStep(p, c, r);
                        return;
                    }
                }
            }
            int block = Best(p, BlockScore);
            if (block < 0)
            {
                Fail("no blocks to pillar up with (need dirt or stone blocks)");
                return;
            }
            if (p.velocity.Y == 0f)
            {
                Hold(p, block);
                Jump(p);
                return;
            }
            Jump(p);
            int floor = feet + 1;
            for (int s = 0; s < 10 && !Solid(px, floor); s++)
                floor++;
            int spot = floor - 1;
            if (spot > feet && Solid(px, floor) && !HasTile(px, spot))
                PlaceStep(p, px, spot, block);
        }

        private static Item NearestDrop(Player p, Func<Item, bool> match, float radiusTiles)
        {
            Item best = null;
            float bestDist = radiusTiles * 16f;
            for (int i = 0; i < Main.maxItems; i++)
            {
                Item drop = Main.item[i];
                if (drop == null || !drop.active || drop.stack <= 0 || !match(drop))
                    continue;
                float d = Vector2.Distance(drop.Center, p.Center);
                if (d < bestDist)
                {
                    bestDist = d;
                    best = drop;
                }
            }
            return best;
        }

        /// <summary>Walks to the nearest matching dropped item so the normal pickup range grabs it.</summary>
        private static bool CollectStep(Player p, Func<Item, bool> match, float radiusTiles)
        {
            Item drop = NearestDrop(p, match, radiusTiles);
            if (drop == null)
                return false;
            StepToward(p, (int)(drop.Center.X / 16f), (int)(drop.Center.Y / 16f) - 1, 0, 2, true, 60);
            return true;
        }

        private static Point? FindTile(Player p, Func<Tile, bool> match, int rx, int ry, bool avoidLava)
        {
            int px = Cx(p), py = Cy(p);
            Point? best = null;
            double bestDist = double.MaxValue;
            for (int dx = -rx; dx <= rx; dx++)
            {
                for (int dy = -ry; dy <= ry; dy++)
                {
                    int x = px + dx, y = py + dy;
                    if (!WorldGen.InWorld(x, y, 10))
                        continue;
                    Tile t = Framing.GetTileSafely(x, y);
                    if (!t.HasTile || !match(t))
                        continue;
                    if (avoidLava && (Lava(x, y - 1) || Lava(x - 1, y) || Lava(x + 1, y) || Lava(x, y + 1)))
                        continue;
                    double d = dx * dx + dy * dy * 1.5;
                    if (d < bestDist)
                    {
                        bestDist = d;
                        best = new Point(x, y);
                    }
                }
            }
            return best;
        }

        // ------------------------------------------------------------------ travel skills

        private static void Goto(Player p)
        {
            int tx = AbsX(ArgI("x", 0)), ty = AbsY(ArgI("y", 0));
            Detail = $"dx={tx - Cx(p)} dy={ty - Cy(p)}";
            if (StepToward(p, tx, ty, 2, ArgI("tol_y", 3), true, 45))
                Done("arrived");
        }

        private static void Explore(Player p)
        {
            int dir = ArgS("dir", "right").StartsWith("l", StringComparison.OrdinalIgnoreCase) ? -1 : 1;
            int tiles = Math.Clamp(ArgI("tiles", 60), 4, 400);
            bool tunnel = ArgS("mode", "walk") == "tunnel";
            int tx = _startX + dir * tiles;
            int px = Cx(p);
            Detail = $"moved {Math.Abs(px - _startX)}/{tiles} tiles";
            if (px < 120 || px > Main.maxTilesX - 120)
            {
                Fail("reached the edge of the world");
                return;
            }
            _wetTicks = p.wet ? _wetTicks + 1 : 0;
            if (_wetTicks > 300)
            {
                Fail("deep water in this direction");
                return;
            }
            if (p.wet)
                Jump(p);
            if (StepToward(p, tx, Cy(p), 1, 9999, true, tunnel ? 3 : 45))
                Done($"moved {tiles} tiles {(dir > 0 ? "right" : "left")}");
        }

        private static void DigDown(Player p)
        {
            int depth = Math.Clamp(ArgI("depth", 20), 2, 300);
            int targetY = _startY + depth;
            int feet = FeetRow(p);
            Detail = $"depth {Cy(p) - _startY}/{depth}";
            if (Cy(p) >= targetY)
            {
                Done($"dug down {Cy(p) - _startY} tiles");
                return;
            }
            for (int c = LeftCol(p); c <= RightCol(p); c++)
            {
                for (int r = feet + 1; r <= feet + 5; r++)
                {
                    if (Lava(c, r))
                    {
                        Fail("lava below; stopped digging");
                        return;
                    }
                }
            }
            if (p.velocity.Y == 0f)
            {
                int air = 0;
                int c0 = Cx(p);
                for (int r = feet + 2; r <= feet + 40 && !Solid(c0, r); r++)
                    air++;
                if (air >= 16)
                {
                    Done($"stopped above a {air}-tile drop into a cave");
                    return;
                }
            }
            StepToward(p, Cx(p), targetY, 9999, 0, true, 0);
        }

        private static void Surface(Player p)
        {
            int px = Cx(p), top = TopRow(p);
            Detail = $"depth below surface {(int)(Cy(p) - Main.worldSurface)}";
            if (p.velocity.Y == 0f && Cy(p) < Main.worldSurface)
            {
                bool open = true;
                for (int r = top - 1; r >= top - 30 && open; r--)
                    for (int c = LeftCol(p); c <= RightCol(p); c++)
                        if (Solid(c, r)) open = false;
                if (open)
                {
                    Done("standing under open sky");
                    return;
                }
            }
            PillarStep(p);
        }

        // ------------------------------------------------------------------ gathering skills

        private static bool IsWood(Item it)
        {
            return it.type == ItemID.Wood || it.type == ItemID.BorealWood || it.type == ItemID.PalmWood
                || it.type == ItemID.RichMahogany || it.type == ItemID.Ebonwood || it.type == ItemID.Shadewood;
        }

        private static void Chop(Player p)
        {
            int want = ArgI("wood", 60);
            int have = Count(p, IsWood);
            Detail = $"wood {have}/{want}";
            if (have >= want)
            {
                Done($"have {have} wood");
                return;
            }
            if (_phase == 1)
            {
                // The tree has fallen: walk over the drops for a few seconds.
                if (_phaseTicks < 420 && CollectStep(p, it => IsWood(it) || it.type == ItemID.Acorn, 24f))
                    return;
                SetPhase(0);
                _hasTarget = false;
            }
            if (!_hasTarget)
            {
                Point? tree = MeteredBridgeSystem.FindNearestTreeBase(p, ArgS("dir", "right").StartsWith("l") ? -1 : 1);
                if (!tree.HasValue)
                {
                    Fail($"no tree within 18 tiles (wood {have}/{want})");
                    return;
                }
                _target = tree.Value;
                _hasTarget = true;
                _approachTicks = 0;
            }
            int res = ApproachMine(p, _target.X, _target.Y);
            if (res == 1)
                SetPhase(1);
        }

        private static void Mine(Player p)
        {
            string kind = ArgS("kind", "stone").ToLowerInvariant();
            int want = ArgI("count", 20);
            Func<Tile, bool> match;
            string itemName;
            if (kind == "stone")
            {
                match = t => t.TileType == TileID.Stone;
                itemName = "Stone Block";
            }
            else if (kind == "life crystal" || kind == "heart")
            {
                match = t => t.TileType == TileID.Heart;
                itemName = "Life Crystal";
            }
            else if (kind == "ore")
            {
                match = t => TileID.Sets.Ore[t.TileType];
                itemName = "";
            }
            else
            {
                match = t => TileID.Sets.Ore[t.TileType]
                    && MeteredBridgeSystem.OreDisplayName(t.TileType).Equals(kind, StringComparison.OrdinalIgnoreCase);
                itemName = kind;
            }
            itemName = ArgS("item", itemName);
            Func<Item, bool> itemMatch = it => itemName == ""
                ? it.Name.EndsWith(" Ore", StringComparison.OrdinalIgnoreCase)
                : string.Equals(it.Name, itemName, StringComparison.OrdinalIgnoreCase);
            int have = Count(p, itemMatch);
            if (Ticks == 1)
                _baseline = have;
            // "count" is how many more to gather in this run.
            Detail = $"{kind} +{have - _baseline}/{want}";
            if (have - _baseline >= want)
            {
                Done($"gathered {have - _baseline} {kind} (now {have})");
                return;
            }
            if (_phase == 1)
            {
                if (_phaseTicks < 180 && CollectStep(p, itemMatch, 10f))
                    return;
                SetPhase(0);
                _hasTarget = false;
            }
            if (!_hasTarget || !HasTile(_target.X, _target.Y))
            {
                Point? found = FindTile(p, match, 34, 24, true);
                if (!found.HasValue)
                {
                    Fail($"no {kind} within 34 tiles (gathered {have - _baseline}/{want})");
                    return;
                }
                _target = found.Value;
                _hasTarget = true;
                _approachTicks = 0;
            }
            int res = ApproachMine(p, _target.X, _target.Y);
            if (res == 1)
            {
                _counter++;
                // Sweep up drops every few tiles rather than after each one.
                if (_counter % 4 == 0)
                    SetPhase(1);
                else
                    _hasTarget = false;
            }
        }

        private static void LootChest(Player p)
        {
            if (_phase == 0)
            {
                if (!_hasTarget)
                {
                    Point? chest = FindTile(p, t => (t.TileType == TileID.Containers || t.TileType == TileID.Containers2)
                        && t.TileFrameX % 36 == 0 && t.TileFrameY % 36 == 0, 34, 24, true);
                    if (!chest.HasValue)
                    {
                        Fail("no chest within 34 tiles");
                        return;
                    }
                    _target = chest.Value;
                    _hasTarget = true;
                }
                if (InReach(p, _target.X, _target.Y, -2) && Math.Abs(_target.X - Cx(p)) <= 3)
                {
                    SetPhase(1);
                    return;
                }
                if (_phaseTicks > 3000)
                {
                    Fail("could not reach the chest");
                    return;
                }
                StepToward(p, _target.X, _target.Y, 0, 0, true, 40);
                return;
            }
            if (_phase == 1)
            {
                if (p.chest != -1)
                {
                    ChestUI.LootAll();
                    SetPhase(2);
                    return;
                }
                if (_phaseTicks > 240)
                {
                    Fail("the chest did not open (locked, or blocked)");
                    return;
                }
                AimTile(_target.X, _target.Y);
                p.controlUseTile = _phaseTicks % 20 == 5; // a right-click every third of a second
                return;
            }
            if (_phaseTicks > 10)
            {
                p.chest = -1;
                Main.playerInventory = false;
                Done("opened the chest and pressed Loot All");
            }
        }

        // ------------------------------------------------------------------ crafting / inventory skills

        private static ushort StationTile(string station)
        {
            switch (station.ToLowerInvariant())
            {
                case "work bench": return TileID.WorkBenches;
                case "furnace": return TileID.Furnaces;
                case "anvil": return TileID.Anvils;
                case "demon altar":
                case "crimson altar":
                case "altar": return TileID.DemonAltar;
                default: return 0;
            }
        }

        private static void Craft(Player p)
        {
            string item = ArgS("item", "");
            int want = Math.Max(1, ArgI("count", 1));
            string station = ArgS("station", "");
            if (item == "")
            {
                Fail("craft needs an item name");
                return;
            }
            if (Ticks == 1)
                _baseline = CountName(p, item);
            int made = CountName(p, item) - _baseline;
            Detail = $"{item} +{made}/{want}";
            if (made >= want)
            {
                Main.playerInventory = false;
                Done($"crafted {item} x{made}");
                return;
            }
            if (_phase == 0)
            {
                ushort tile = station == "" ? (ushort)0 : StationTile(station);
                if (tile != 0)
                {
                    if (!_hasTarget)
                    {
                        Point? s = FindTile(p, t => t.TileType == tile, 40, 24, false);
                        if (!s.HasValue)
                        {
                            Fail($"no {station} within 40 tiles");
                            return;
                        }
                        _target = s.Value;
                        _hasTarget = true;
                    }
                    if (!StepToward(p, _target.X, _target.Y - 1, 2, 2, true, 45))
                    {
                        if (_phaseTicks > 2400)
                            Fail($"could not reach the {station}");
                        return;
                    }
                }
                SetPhase(1);
                return;
            }
            // Standing at the station: one craft click every fifth of a second.
            if (_phaseTicks % 12 != 6)
                return;
            var sys = ModContent.GetInstance<MeteredBridgeSystem>();
            if (!sys.TryCraft(p, item))
            {
                Main.playerInventory = false;
                if (made > 0)
                    Done($"crafted {item} x{made} then ran out of materials (wanted {want})");
                else
                    Fail($"recipe for {item} not available (missing materials or not at the crafting station)");
            }
        }

        private static void Equip(Player p)
        {
            string item = ArgS("item", "");
            int before = CountName(p, item);
            if (item == "" || before == 0)
            {
                Fail($"'{item}' is not in the inventory");
                return;
            }
            ModContent.GetInstance<MeteredBridgeSystem>().TryUseInventoryItem(p, item);
            Main.playerInventory = false;
            if (CountName(p, item) < before)
                Done($"equipped {item}");
            else
                Fail($"{item} could not be equipped (not armor/accessory, or no free slot)");
        }

        private static void UseItemSkill(Player p)
        {
            string item = ArgS("item", "");
            int slot = SlotByName(p, item);
            int have = CountName(p, item);
            if (Ticks == 1)
                _baseline = have;
            bool isSummon = item.Equals("Suspicious Looking Eye", StringComparison.OrdinalIgnoreCase);
            if (have < _baseline || (isSummon && NPC.AnyNPCs(NPCID.EyeofCthulhu)))
            {
                Done($"used {item}");
                return;
            }
            if (slot < 0)
            {
                Fail($"'{item}' is not in the inventory");
                return;
            }
            if (_phase == 1 && p.itemAnimation == 0 && _phaseTicks > 20 && !p.inventory[p.selectedItem].consumable)
            {
                Done($"used {item}");
                return;
            }
            if (Ticks > 240)
            {
                Fail(isSummon
                    ? "Suspicious Looking Eye did nothing (it only works at night with no Eye already present)"
                    : $"{item} could not be used right now");
                return;
            }
            Aim(p.Center + new Vector2(p.direction * 48f, -24f));
            if (Hold(p, slot))
            {
                Use(p);
                if (p.itemAnimation > 0 && _phase == 0)
                    SetPhase(1);
            }
        }

        private static void Place(Player p)
        {
            string item = ArgS("item", "");
            int slot = SlotByName(p, item);
            if (slot < 0)
                slot = Best(p, it => it.createTile >= 0 && MeteredBridgeSystem.ItemMatchesQuery(it, item) ? 1 : 0);
            if (slot < 0)
            {
                if (_phase > 0)
                    Done($"placed {item}");
                else
                    Fail($"'{item}' is not in the inventory");
                return;
            }
            Item it0 = p.inventory[slot];
            if (it0.createTile < 0)
            {
                Fail($"{item} is not placeable");
                return;
            }
            int have = Count(p, it => it.type == it0.type);
            if (Ticks == 1)
                _baseline = have;
            if (have < _baseline)
            {
                Done($"placed {it0.Name}");
                return;
            }
            int w = 1, h = 1, ox = 0, oy = 0;
            TileObjectData data = TileObjectData.GetTileData(it0.createTile, it0.placeStyle);
            if (data != null)
            {
                w = data.Width;
                h = data.Height;
                ox = data.Origin.X;
                oy = data.Origin.Y;
            }
            int feet = FeetRow(p), px = Cx(p);
            // Candidate spots on the floor the player stands on, nearest first, alternating sides.
            int tried = 0;
            for (int k = 0; k < 10; k++)
            {
                int off = 2 + k / 2;
                int left = (k % 2 == 0) ? px + off : px - off - (w - 1);
                bool ok = true;
                for (int c = left; c < left + w && ok; c++)
                {
                    if (!Solid(c, feet + 1)) ok = false;
                    for (int r = feet - h + 1; r <= feet && ok; r++)
                        if (HasTile(c, r)) ok = false;
                }
                if (!ok || !InReach(p, left + ox, feet - h + 1 + oy, 0))
                    continue;
                if (tried++ < _counter)
                    continue;
                int res = PlaceStep(p, left + ox, feet - h + 1 + oy, slot);
                if (res == 1)
                    SetPhase(1);
                else if (res == -1)
                    _counter++;
                return;
            }
            Fail($"no flat free spot {w} wide on this floor within reach to place {it0.Name}; move a few tiles and retry");
        }

        // ------------------------------------------------------------------ arena

        /// <summary>
        /// Levels a runway at the height the player started on: mines whatever stands in the three body rows and
        /// fills holes in the floor row, one column at a time, walking from the left end to the right end.
        /// </summary>
        private static void BuildArena(Player p)
        {
            int width = Math.Clamp(ArgI("width", 80), 20, 200);
            if (Ticks == 1)
            {
                _target = new Point(_startX, FeetRow(p) + 1); // x centre, floor row
                _counter = _startX - width / 2;               // next column to finish
            }
            int floor = _target.Y;
            int endX = _target.X + width / 2;
            Detail = $"column {_counter - (_target.X - width / 2)}/{width}";
            if (_counter > endX)
            {
                Done($"runway {width} tiles wide levelled at this height");
                return;
            }
            int col = _counter;
            if (!InReach(p, col, floor - 1, -1) || Math.Abs(col - Cx(p)) > 3)
            {
                if (++_approachTicks > 1800)
                {
                    Fail("could not reach the next runway column");
                    return;
                }
                StepToward(p, col - 2, floor - 2, 0, 1, true, 30);
                return;
            }
            _approachTicks = 0;
            for (int r = floor - 4; r <= floor - 1; r++)
            {
                Tile t = Framing.GetTileSafely(col, r);
                if (t.HasTile && (Solid(col, r) || MeteredBridgeSystem.IsTreeTile(t)))
                {
                    MineStep(p, col, r);
                    return;
                }
            }
            if (!HasTile(col, floor))
            {
                // Standing in the column would block the placement; step back first.
                if (col >= LeftCol(p) && col <= RightCol(p) && FeetRow(p) >= floor)
                {
                    p.controlLeft = true;
                    return;
                }
                int slot = Best(p, it => it.createTile == TileID.Platforms ? 200 : BlockScore(it));
                if (slot < 0)
                {
                    Fail("no blocks or platforms left to fill the runway floor");
                    return;
                }
                if (PlaceStep(p, col, floor, slot) == -1)
                    _counter++; // nothing to attach to here; leave the gap
                return;
            }
            _counter++;
        }

        // ------------------------------------------------------------------ combat

        private static bool HasAmmoFor(Player p, Item weapon)
        {
            for (int i = 0; i < 58; i++)
            {
                Item it = p.inventory[i];
                if (it != null && !it.IsAir && it.stack > 0 && it.ammo == weapon.useAmmo)
                    return true;
            }
            return false;
        }

        private static int RangedScore(Player p, Item it)
        {
            return it.damage > 0 && it.useAmmo > 0 && HasAmmoFor(p, it) ? it.damage + 1 : 0;
        }

        private static int MeleeScore(Item it)
        {
            if (it.damage <= 0 || it.useAmmo > 0 || it.mana > 0 || it.consumable || it.createTile >= 0 || it.createWall > 0)
                return 0;
            // Tools can swing too, but only as a last resort.
            if (it.pick > 0 || it.axe > 0 || it.hammer > 0)
                return 1;
            return it.damage + 1;
        }

        private static NPC NearestHostile(Player p, float radiusTiles, string nameFilter)
        {
            NPC best = null;
            float bestDist = radiusTiles * 16f;
            for (int i = 0; i < Main.maxNPCs; i++)
            {
                NPC n = Main.npc[i];
                if (n == null || !n.active || n.friendly || n.townNPC || n.life <= 0 || n.damage <= 0 || n.dontTakeDamage)
                    continue;
                if (nameFilter != null && n.TypeName.IndexOf(nameFilter, StringComparison.OrdinalIgnoreCase) < 0)
                    continue;
                float d = Vector2.Distance(n.Center, p.Center);
                if (d < bestDist)
                {
                    bestDist = d;
                    best = n;
                }
            }
            return best;
        }

        /// <summary>
        /// One step of fighting a target with whatever is in the inventory: best bow/gun that has ammo, otherwise the
        /// best melee weapon. Aims with the mouse (leading a moving target), holds the attack button, runs away along
        /// the runway when fleeing, jumps over charges, and drinks a potion with the quick-heal key when low.
        /// </summary>
        private static void CombatStep(Player p, NPC t, bool flee, int minX, int maxX)
        {
            float dist = Vector2.Distance(t.Center, p.Center);
            int ranged = Best(p, it => RangedScore(p, it));
            int melee = Best(p, MeleeScore);
            bool useRanged = ranged >= 0 && (flee || dist > 72f || melee < 0);
            int slot = useRanged ? ranged : melee;
            Vector2 aim = t.Center;
            if (useRanged)
            {
                float speed = Math.Max(6f, p.inventory[ranged].shootSpeed + 3f);
                float flight = Math.Min(60f, dist / speed);
                aim += t.velocity * flight;
                aim.Y -= dist * 0.04f; // arrows drop
            }
            Aim(aim);
            if (slot >= 0 && Hold(p, slot))
                Use(p);

            int px = Cx(p);
            float adx = Math.Abs(t.Center.X - p.Center.X);
            int moveDir = 0;
            if (flee)
            {
                int away = p.Center.X >= t.Center.X ? 1 : -1;
                if (_runDir == 0)
                    _runDir = away;
                if (_edgeHold > 0)
                    _edgeHold--;
                else if (adx > 160f)
                    _runDir = away;
                if (px >= maxX) { _runDir = -1; _edgeHold = 150; }
                if (px <= minX) { _runDir = 1; _edgeHold = 150; }
                moveDir = _runDir;
            }
            else if (!useRanged && adx > 28f)
            {
                moveDir = t.Center.X > p.Center.X ? 1 : -1;
            }
            if (moveDir > 0) p.controlRight = true;
            if (moveDir < 0) p.controlLeft = true;

            bool blockedAhead = false;
            if (moveDir != 0)
            {
                int col = moveDir > 0 ? (int)((p.position.X + p.width + 3f) / 16f) : (int)((p.position.X - 3f) / 16f);
                for (int r = TopRow(p); r <= FeetRow(p); r++)
                    if (Solid(col, r)) blockedAhead = true;
            }
            Vector2 toPlayer = p.Center - t.Center;
            bool charging = t.velocity.Length() > 5f && dist < 260f && Vector2.Dot(t.velocity, toPlayer) > 0f;
            if (blockedAhead || charging)
                Jump(p);

            if (p.statLife < p.statLifeMax2 / 2 && !p.HasBuff(BuffID.PotionSickness) && Ticks % 12 == 0)
                p.controlQuickHeal = true;
        }

        private static void Fight(Player p)
        {
            NPC boss = MeteredBridgeSystem.FindBoss();
            if (Ticks == 1)
            {
                _baseline = NPC.downedBoss1 ? 1 : 0;
                _counter = 0;
            }
            if (boss == null)
            {
                if (NPC.downedBoss1 && (_counter > 0 || _baseline == 0))
                {
                    Done("Eye of Cthulhu defeated");
                    return;
                }
                if (_counter > 0)
                {
                    Fail("the boss left before it was killed (dawn, or it flew away)");
                    return;
                }
                if (Ticks > 900)
                {
                    Fail("no boss appeared within 15s; summon it first");
                    return;
                }
                NPC near = NearestHostile(p, 14f, null);
                if (near != null)
                    CombatStep(p, near, false, 0, 0);
                return;
            }
            _counter++;
            int minX = HasArg("x_min") ? AbsX(ArgI("x_min", 0)) : _startX - 45;
            int maxX = HasArg("x_max") ? AbsX(ArgI("x_max", 0)) : _startX + 45;
            Detail = $"{boss.TypeName} {boss.life}/{boss.lifeMax} hp, player {p.statLife}/{p.statLifeMax2}";
            // Servants that get close are shot first; they die in a hit or two and drop hearts.
            NPC target = boss;
            NPC add = NearestHostile(p, 7f, null);
            if (add != null && !add.boss)
                target = add;
            CombatStep(p, target, true, minX, maxX);
        }

        private static void Hunt(Player p)
        {
            string npcName = ArgS("npc", "Demon Eye");
            string dropName = ArgS("item", "Lens");
            int want = ArgI("count", 6);
            int have = CountName(p, dropName);
            Detail = $"{dropName} {have}/{want}";
            if (have >= want)
            {
                Done($"have {have} {dropName}");
                return;
            }
            if (Main.dayTime && ArgS("when", "night") == "night")
            {
                Fail($"it is day; {npcName} only comes out at night ({dropName} {have}/{want})");
                return;
            }
            NPC threat = NearestHostile(p, 9f, null);
            NPC prey = NearestHostile(p, 45f, npcName);
            NPC target = threat ?? prey;
            Func<Item, bool> dropMatch = it => string.Equals(it.Name, dropName, StringComparison.OrdinalIgnoreCase);
            if (threat == null && CollectStep(p, dropMatch, 50f))
                return;
            if (target != null)
                CombatStep(p, target, false, 0, 0);
        }

        private static void Wait(Player p)
        {
            string until = ArgS("until", "night").ToLowerInvariant();
            Detail = Main.dayTime ? "day" : "night";
            if ((until == "night" && !Main.dayTime) || (until == "day" && Main.dayTime))
            {
                Done("it is now " + until);
                return;
            }
            NPC threat = NearestHostile(p, 12f, null);
            if (threat != null)
                CombatStep(p, threat, false, 0, 0);
        }
    }
}
