# MuJoCo Starter Project

這是一個適用於 Windows 與 MuJoCo 3.11.0 的入門專案，包含：

- 可直接載入的 MJCF 場景
- Python 控制範例
- Windows 一鍵啟動腳本
- 基本安裝與執行說明

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
├─ models/
│  └─ falling_box.xml
├─ src/
│  └─ run_simulation.py
├─ .gitignore
├─ requirements.txt
├─ run_windows.bat
└─ README.md
```

## 5. 操作方式

模型載入後：

- 按 `Space`：開始或暫停模擬
- 滑鼠左鍵拖曳：旋轉視角
- 滑鼠右鍵拖曳：平移視角
- 滾輪：縮放

場景包含一個地面與一個可自由落下的紅色方塊，可作為後續雙臂機器人、工料搬運與碰撞測試的基礎。
