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
        self.profile_values = []

    def request_viewer_open(self):
        self.viewer_requests += 1
        return "requested"

    def set_paused(self, paused):
        self.paused = paused

    def update_settings(self, values):
        self.settings_values = values

    def save_profile(self, key, concept=""):
        self.profile_values.append(("save", key, concept))

    def load_profile(self, key):
        self.profile_values.append(("load", key))

    def delete_profile(self, key):
        self.profile_values.append(("delete", key))

    def snapshot(self):
        return {
            "viewer": {"active": False, "launching": False, "requested": self.viewer_requests > 0},
            "profile": {"key": "test_key", "concept": "test concept"},
            "profile_library": [{"key": "test_key", "name": "測試", "concept": "test concept"}],
        }


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
        self.assertIn("低頻運行觀測", html)
        self.assertIn("共享皮帶區閘門", html)
        self.assertIn('id="shared-zone"', html)
        self.assertIn("離帶後 Peer 時序", html)
        self.assertIn('id="collision-priority"', html)
        self.assertIn("最低運行優先", html)
        self.assertIn('name="profile_key"', html)
        self.assertIn('id="save-profile"', html)
        self.assertIn('id="load-profile"', html)
        self.assertIn('id="delete-profile"', html)
        self.assertIn('id="profile-select"', html)
        self.assertIn('id="profile-concept"', html)
        self.assertIn('id="profile-key-list"', html)
        self.assertIn('id="profile-summary"', html)

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

    def test_profile_save_and_load_actions_reach_the_simulator(self):
        demo = FakeDemo()
        dashboard = start_dashboard(demo, port=0)
        try:
            for action in ("save_profile", "load_profile", "delete_profile"):
                request = Request(
                    f"{dashboard.url}/api/control",
                    data=json.dumps({"action": action, "values": {"profile_key": "test_key", "profile_concept": "test concept"}}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=2):
                    pass
            self.assertEqual(demo.profile_values, [("save", "test_key", "test concept"), ("load", "test_key"), ("delete", "test_key")])
        finally:
            dashboard.stop()


if __name__ == "__main__":
    unittest.main()
