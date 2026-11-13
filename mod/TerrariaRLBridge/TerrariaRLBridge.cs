using System;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using Terraria;
using Terraria.ModLoader;

namespace TerrariaRLBridge
{
    public class TerrariaRLBridgeMod : Mod
    {
    }

    /// <summary>
    /// Thin tModLoader ModSystem hook that freezes world time between agent
    /// operations and forwards player state / macro commands to the pure-Python
    /// bridge server running at http://127.0.0.1:8765/.
    /// </summary>
    public class MeteredBridgeSystem : ModSystem
    {
        private static readonly HttpClient Http = new HttpClient { Timeout = TimeSpan.FromMilliseconds(250) };
        private const string BridgeUrl = "http://127.0.0.1:8765/";
        private const int TicksPerWindow = 27000;

        public bool MeteredAgentMode = true;
        public int PendingBurstTicks;
        public int WindowTicksElapsed;
        private double _frozenTime;
        private bool _frozenDayTime = true;

        public override void OnWorldLoad()
        {
            PendingBurstTicks = 0;
            WindowTicksElapsed = 0;
            _frozenTime = Main.time;
            _frozenDayTime = Main.dayTime;
        }

        public override void PostUpdateEverything()
        {
            if (Main.dedServ || Main.gameMenu || !MeteredAgentMode)
                return;

            if (PendingBurstTicks <= 0)
            {
                // Freeze world progression while awaiting the next Python MCP operation
                Main.time = _frozenTime;
                Main.dayTime = _frozenDayTime;
                PollBridgeStatus();
                return;
            }

            PendingBurstTicks--;
            WindowTicksElapsed = Math.Min(TicksPerWindow, WindowTicksElapsed + 1);
            _frozenTime = Main.time;
            _frozenDayTime = Main.dayTime;
        }

        private void PollBridgeStatus()
        {
            try
            {
                var p = Main.LocalPlayer;
                var payload = JsonSerializer.Serialize(new
                {
                    jsonrpc = "2.0",
                    id = WindowTicksElapsed,
                    method = "tools/call",
                    @params = new
                    {
                        name = "terraria_status",
                        arguments = new { x = (int)(p.position.X / 16f), y = (int)(p.position.Y / 16f), hp = p.statLife }
                    }
                });
                using var content = new StringContent(payload, Encoding.UTF8, "application/json");
                _ = Http.PostAsync(BridgeUrl, content);
            }
            catch
            {
                // Standalone headless mode runs in Python when live game bridge is inactive.
            }
        }
    }
}
