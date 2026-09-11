# MuJoCo Nova5 Spatiotemporal Planning Project

本專案以 MuJoCo 為數位孿生平台，逐步建立雙 DOBOT Nova5 的三維時空 voxel 運動規劃系統，並以後續可擴展至多機械臂為設計方向。

目前研究主線：

```text
Task
→ Multiple Motion Candidates
→ Kinematic Filtering
→ Trajectory Generation
→ Future Spatiotemporal Occupancy
→ Collision Filtering
→ Best Safe Motion
```

共享工作區使用 `5 × 5 × 5 = 125` 個 voxel，後續會把每條候選 trajectory 轉換成 future occupancy，依「相同空間 + 重疊時間」判定 Spatiotemporal Conflict。

完整術語定義、演算法架構與開發階段請見：

- [`docs/spatiotemporal_voxel_planning.md`](docs/spatiotemporal_voxel_planning.md)

## 目前環境

這是一個適用於 Windows 與 MuJoCo 3.11.0 的專案，包含：

- 可直接載入的 MJCF 場景
- Python 控制範例
- Windows 一鍵啟動腳本
- MuJoCo / Nova5 時空規劃研究文件

## 1. 下載專案

```bat
git clone https://github.com/hanpapera0202/mujoco.git
cd mujoco
```

## 2. 使用 MuJoCo `simulate.exe` 執行

先確認 MuJoCo 已解壓縮，例如：

```text
C:\mujoco\mujoco-3.11.0-windows-x86_64
```

若路徑不同，請修改 `run_windows.bat` 內的 `MUJOCO_HOME`。

雙擊：

```text
run_windows.bat
```

或在命令提示字元執行：

```bat
run_windows.bat
```

## 3. 使用 Python 執行

建立虛擬環境：

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

執行：

```bat
python src\run_simulation.py
```

## 4. 專案結構

```text
mujoco/
├─ docs/
│  └─ spatiotemporal_voxel_planning.md
├─ models/
├─ src/
├─ .gitignore
├─ requirements.txt
├─ run_windows.bat
└─ README.md
```

## 5. MuJoCo 操作方式

模型載入後：

- 按 `Space`：開始或暫停模擬
- 滑鼠左鍵拖曳：旋轉視角
- 滑鼠右鍵拖曳：平移視角
- 滾輪：縮放
