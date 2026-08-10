# Nova5 Dual-Arm Sorting Research

Nova5 雙手臂流水線分揀的 MuJoCo 場景與中央協調演算法第一版。

目前版本的目的，是在相同的 MuJoCo / IK 執行器前提下比較高階任務分派方法。中央協調器會快速排除不可行候選，再以全域方式指派兩台平等的手臂，並以同步軌跡與碰撞領域檢查任務。

## 目前規則

- 30 秒滾動規劃視窗，兩台平權手臂每次最多各有一個未完成預約。
- 固定 seed 的 10 件物體全部是 `MIDDLE`，不再先貼 LEFT / RIGHT 標籤。
- 中央端以可達性、期限、成功率、路徑成本與歷史負載選擇 A 或 B；真正完全同分時首輪 A 優先。
- 每臂以上臂、前臂、夾爪三個長方體建模；雙臂同時活動時，以盒體各面外擴 10 cm 的領域預警。
- 固定環境碰撞會立即撤銷候選並重新匹配；雙臂領域衝突則保留預約等待安全窗口。

## 執行

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 開啟 MuJoCo 流水線
python src\run_sorting_line.py

# 開啟固定 seed 的雙手臂抓取、分揀與本機 Web 控制台

目前版本：`0.5.2`。版本與 Git 推送規範見 [versioning_zh.md](docs/versioning_zh.md)。
python src\run_sorting_demo.py --seed 42

# 執行可重現的第一版基準範例
python src\run_benchmark.py --seeds 30 --output-dir results\v1
```

基準輸出包含 `events.jsonl`（每次決策與結果）與 `metrics.csv`（漏件率、正確分流率、平均取件時間、近失次數、雙臂同時工作比例）。目前基準使用固定時間模型；下一階段會讓 `run_sorting_line.py` 回傳 MuJoCo 實測事件。

## MuJoCo 抓取演示

`src/run_sorting_demo.py` 是建置展示用的 10 件連續投料場景。皮帶有效寬度為 45 cm，固定 seed 的全部工件都從中央共享帶進入，皮帶速度為 `0.12 m/s`。兩台 Nova5 基座位於 x = +/-0.58 m。中央協調器安排任務後，手臂以 6D 姿態 IK 依序執行接近、下降、閉合、抬升、移至托盤、放開與回原位。只有 MuJoCo 回報雙側指墊實體接觸才算抓取成功，不使用 weld/equality constraint，且放置必須由工件最後落入目標托盤驗證。

目前演算法名稱為 **CSPR（Centralized Spatiotemporal Reservation，集中式時空預約）**。執行器會以實測週期回授更新派工估計、限制投料負載，並在相同時間軸上預檢候選與既有手臂的關節路徑。抓取需雙側指墊同時接觸，且不使用 weld；最後仍須通過托盤位置驗證才列為成功。控制台已保留 Deadline-first、Hungarian、Fuzzy 的切換位置，目前只啟用 CSPR。

啟動演示後，瀏覽器會開啟 `http://127.0.0.1:8765`。控制台可開始、暫停、重播、開啟或重新顯示 MuJoCo、修改 seed、調整皮帶/演示速度、調整中央協調器參數，並查看 A/B 任務、最新派工、拒絕原因與事件紀錄。關閉 MuJoCo 視窗後，控制服務仍會保留同一場演示，按「開始 / 繼續」即可重新開啟。MuJoCo 視窗聚焦後按 `R` 也會重播。

演算法的數學定義、程式對照與參數修改說明位於 [docs/algorithm_math_zh.md](docs/algorithm_math_zh.md)。

## 專案記憶與防退化

新增需求前先在 [專案記憶與驗收契約](docs/project_memory_zh.md) 建立需求 ID，再修改程式並加入可量測的測試。根目錄的 `AGENTS.md` 會提醒後續 Codex 工作先讀取這份台帳，並禁止以放寬判定、啟用 weld 或刪除測試掩蓋退化。

每次修改後執行：

```powershell
python -m unittest discover -s tests -v
```

固定 seed 測試會實際運行 MuJoCo 至第一件工件完成放置，檢查中央派工、雙指實體接觸、無 equality constraint、10 cm 警戒模型與無安全失敗。另有單元測試驗證兩件 MIDDLE 可同時分配給平權雙臂。GitHub Actions 會在每次 push 或 pull request 再執行一次相同門檻。
