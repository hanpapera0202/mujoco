# Nova5 Dual-Arm Sorting Research

Nova5 雙手臂流水線分揀的 MuJoCo 場景與中央協調演算法第一版。

目前版本的目的，是在相同的 MuJoCo / IK 執行器前提下比較高階任務分派方法。中央協調器會快速排除不可行候選，再以全域方式指派兩台平等的手臂，並以同步軌跡與碰撞領域檢查任務。

## 目前規則

- 30 秒滾動規劃視窗，兩台平權手臂每次最多各有一個未完成預約。
- 固定 seed 的 10 件物體全部是 `MIDDLE`，不再先貼 LEFT / RIGHT 標籤。
- 中央端以可達性、期限、成功率、路徑成本與歷史負載選擇 A 或 B；真正完全同分時首輪 A 優先。
- 每臂以上臂、前臂、三段腕部與夾爪六個長方體建模；視覺 mesh 不參與碰撞，雙臂同時活動時以盒體各面外擴 10 cm 的領域預警。
- 固定環境碰撞會立即撤銷候選並重新匹配；雙臂領域衝突則保留預約等待安全窗口。

## 執行

### VS Code（建議）

1. 在 VS Code 選擇「開啟資料夾」，開啟 `C:\Users\Han\PycharmProjects\mujoco`。
2. 按 `Ctrl+Shift+P`，執行「Tasks: Run Task」，選擇「初始化 Python 與 MuJoCo 環境」。首次會建立 `.venv`、安裝 MuJoCo 與 Python 擴充功能。
3. 按 `F5`，從上方啟動選單選擇「雙臂分揀演示（GUI + MuJoCo）」或「單臂抓取驗證（GUI + MuJoCo）」。啟動後會開啟 MuJoCo 與 `http://127.0.0.1:8765` 控制台。

共享的 VS Code 啟動、測試與 Python 設定位於 `.vscode/`；它們會跟著版本控制，讓每台電腦使用相同的執行方式。

### PowerShell

```powershell
.\scripts\setup_windows.ps1

# 開啟 MuJoCo 流水線
& .\.venv\Scripts\python.exe src\run_sorting_line.py

# 開啟固定 seed 的雙手臂抓取、分揀與本機 Web 控制台
& .\.venv\Scripts\python.exe src\run_sorting_demo.py --seed 42 --duration 3600

# 執行可重現的第一版基準範例
& .\.venv\Scripts\python.exe src\run_benchmark.py --seeds 30 --output-dir results\v1
```

目前基準版本：`0.26.1`。版本與 Git 推送規範見 [versioning_zh.md](docs/versioning_zh.md)。單臂場景位於 `models/nova5/nova5_single_arm_sorting_line.xml`，可用 `src/run_single_arm_demo.py` 啟動。最近一次 100-seed 物理驗證與改進方向記錄於 [dual_arm_validation_20260816_zh.md](docs/dual_arm_validation_20260816_zh.md)，本次抓取與碰撞建模驗證見 [v0.25_validation_zh.md](docs/v0.25_validation_zh.md)。

基準輸出包含 `events.jsonl`（每次決策與結果）與 `metrics.csv`（漏件率、正確分流率、平均取件時間、近失次數、雙臂同時工作比例）。目前基準使用固定時間模型；下一階段會讓 `run_sorting_line.py` 回傳 MuJoCo 實測事件。

## MuJoCo 抓取演示

`src/run_sorting_demo.py` 是建置展示用的 10 件連續投料場景。皮帶有效寬度為 45 cm，固定 seed 的全部工件都從中央共享帶進入，皮帶速度為 `0.12 m/s`；每批預設同時投放 2 件，GUI 可調整批次數量及投料 X/Y 最小、最大值。兩台 Nova5 基座位於 x = +/-0.58 m。中央協調器安排任務後，手臂以 6D 姿態 IK 依序執行提前準備、跟帶靠近、下降、閉合、抬升、移至托盤、放開與回原位。物體狀態以 500 Hz 讀取，IK 以熱啟動 25 Hz 更新；閉爪期間仍沿皮帶追蹤。只有 MuJoCo 回報雙側指墊實體接觸才算抓取成功，不使用 weld/equality constraint，且放置必須由工件最後落入目標托盤驗證。

目前演算法名稱為 **BC-GP-JSP（Bayesian Centralized Genetic-Particle Joint Strategy Planner，貝式集中遺傳粒子聯合策略規劃）**。CSPR 保留為低成本硬條件快篩；GA 選擇雙臂離散任務與路徑組合，PSO 微調平行運動權重，再由原有 QP-RRIK 執行。原有 9 組聯合策略與碰撞預檢仍保留，控制台可調碰撞警戒盒的外擴距離，預設仍為 10 cm；實體碰撞盒不會隨之縮放。

啟動演示後，瀏覽器會開啟 `http://127.0.0.1:8765`。控制台可開始、暫停、重播、開啟或重新顯示 MuJoCo、修改 seed、調整皮帶/演示速度、碰撞警戒外擴距離與中央協調器參數，並查看 9 組聯合策略的選擇結果、後驗完工機率、A/B 任務、拒絕原因與事件紀錄。關閉 MuJoCo 視窗後，控制服務仍會保留同一場演示，按「開始 / 繼續」即可重新開啟。MuJoCo 視窗聚焦後按 `R` 也會重播。

演算法的數學定義、程式對照與參數修改說明位於 [docs/algorithm_math_zh.md](docs/algorithm_math_zh.md)。

## 專案記憶與防退化

新增需求前先在 [專案記憶與驗收契約](docs/project_memory_zh.md) 建立需求 ID，再修改程式並加入可量測的測試。根目錄的 `AGENTS.md` 會提醒後續 Codex 工作先讀取這份台帳，並禁止以放寬判定、啟用 weld 或刪除測試掩蓋退化。

每次修改後執行：

```powershell
python -m unittest discover -s tests -v
```

固定 seed 測試會實際運行 MuJoCo 至第一件工件完成放置，檢查中央派工、雙指實體接觸、無 equality constraint、10 cm 警戒模型與無安全失敗。另有單元測試驗證兩件 MIDDLE 可同時分配給平權雙臂。GitHub Actions 會在每次 push 或 pull request 再執行一次相同門檻。
