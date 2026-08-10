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

    def request_viewer_open(self):
        self.viewer_requests += 1
        return "requested"

    def set_paused(self, paused):
        self.paused = paused

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


if __name__ == "__main__":
    unittest.main()
