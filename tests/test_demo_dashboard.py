import json
from pathlib import Path
import sys
import unittest
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demo_dashboard import start_dashboard


class FakeDemo:
    def __init__(self):
        self.viewer_requests = 0
        self.paused = True
        self.settings_values = None

    def request_viewer_open(self):
        self.viewer_requests += 1
        return "requested"

    def set_paused(self, paused):
        self.paused = paused

    def update_settings(self, values):
        self.settings_values = values

    def snapshot(self):
        return {"viewer": {"active": False, "launching": False, "requested": self.viewer_requests > 0}}


class DashboardViewerControlTests(unittest.TestCase):
    def test_open_mujoco_action_reaches_the_corresponding_demo(self):
        demo = FakeDemo()
        dashboard = start_dashboard(demo, port=0)
        try:
            request = Request(
                f"{dashboard.url}/api/control",
                data=json.dumps({"action": "open_mujoco"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=2) as response:
                state = json.load(response)
            self.assertEqual(demo.viewer_requests, 1)
            self.assertTrue(state["viewer"]["requested"])
        finally:
            dashboard.stop()

    def test_start_resumes_and_opens_the_corresponding_mujoco(self):
        demo = FakeDemo()
        dashboard = start_dashboard(demo, port=0)
        try:
            request = Request(
                f"{dashboard.url}/api/control",
                data=json.dumps({"action": "start"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=2):
                pass
            self.assertFalse(demo.paused)
            self.assertEqual(demo.viewer_requests, 1)
        finally:
            dashboard.stop()

    def test_chinese_dashboard_exposes_the_viewer_button(self):
        html = (ROOT / "src" / "dashboard" / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="open-mujoco"', html)
        self.assertIn("開啟 / 顯示 MuJoCo", html)

    def test_visible_demo_settings_action_reaches_the_simulator(self):
        html = (ROOT / "src" / "dashboard" / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="apply-demo-settings"', html)
        self.assertLess(html.index('id="apply-demo-settings"'), html.index('<form id="settings">'))
        demo = FakeDemo()
        dashboard = start_dashboard(demo, port=0)
        try:
            values = {"feed_interval_s": 2.0, "belt_speed_mps": 0.12}
            request = Request(
                f"{dashboard.url}/api/control",
                data=json.dumps({"action": "settings", "values": values}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=2):
                pass
            self.assertEqual(demo.settings_values, values)
        finally:
            dashboard.stop()


if __name__ == "__main__":
    unittest.main()
