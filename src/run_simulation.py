from __future__ import annotations

from pathlib import Path
import time

import mujoco
import mujoco.viewer


def main() -> None:
    """Load the starter MJCF model and run it in MuJoCo's passive viewer."""
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "falling_box.xml"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    print(f"MuJoCo version: {mujoco.__version__}")
    print(f"Model: {model_path}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print("Close the viewer window to stop the program.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_started = time.perf_counter()

            mujoco.mj_step(model, data)
            viewer.sync()

            remaining = model.opt.timestep - (time.perf_counter() - step_started)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
