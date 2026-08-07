# Nova5 Dual-Arm Sorting Research

Nova5 雙手臂流水線分揀的 MuJoCo 場景與中央協調演算法第一版。

第一版的目的不是直接控制關節，而是在相同的 MuJoCo / IK 執行器前提下，比較不同高階任務分派方法。中央協調器會快速排除不可行候選，再以全域方式指派兩台平等的手臂，並用「工作區 + 時間區間」建立不可撤銷的預約。

## 目前規則

- 5 秒滾動規劃視窗，兩台手臂每次最多各有一個未完成預約。
- `LEFT` 只交給 Robot A，放入左側區域；`RIGHT` 只交給 Robot B，放入右側區域。
- `MIDDLE` 位於共享工作區，中央端同時選擇取件手臂與左/右放置區。
- 同一共享區時段只能有一台手臂進入；未獲得中間件的手臂會優先處理自己的專屬區任務。
- 預約一旦建立不重新指派；物件超過尾端而未取件即為 `MISSED`。

## 執行

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 開啟 MuJoCo 流水線
python src\run_sorting_line.py

# 開啟固定 seed 的雙手臂抓取、分揀與本機 Web 控制台
python src\run_sorting_demo.py --seed 42

# 執行可重現的第一版基準範例
python src\run_benchmark.py --seeds 30 --output-dir results\v1
```

基準輸出包含 `events.jsonl`（每次決策與結果）與 `metrics.csv`（漏件率、正確分流率、平均取件時間、近失次數、雙臂同時工作比例）。目前基準使用固定時間模型；下一階段會讓 `run_sorting_line.py` 回傳 MuJoCo 實測事件。

## MuJoCo 抓取演示

`src/run_sorting_demo.py` 是建置展示用的 10 件連續投料場景。固定 seed 先產生 `LEFT` 與 `RIGHT` 工件以供雙臂並行，再交錯產生共享區 `MIDDLE` 工件；皮帶速度為 `0.24 m/s`。中央協調器安排共享區與專屬區任務，再讓兩台 Nova5 以 6D 姿態 IK 維持水平雙指、垂直指面的抓取姿態，依序執行接近、下降、夾爪閉合、抬升、移至實體托盤、放開與回原位。只有 MuJoCo 回報指墊與工件的實體接觸才會啟用夾持約束；接觸前工件完全由輸送帶物理運動，放開時立即解除約束，且放置必須由工件最後落入目標托盤的範圍驗證。

目前演算法名稱為 **CSPR（Centralized Spatiotemporal Reservation，集中式時空預約）**。若另一手臂正在中央走廊的接近、下降、閉合或抬升階段，`MIDDLE` 任務會先保持預約並在控制台顯示「安全等待」，等中央走廊淨空才啟動；MuJoCo 另有 A/B 接觸偵測，偵測到跨手臂接觸會立即暫停。控制台已保留 Deadline-first、Hungarian、Fuzzy 的切換位置，目前只啟用 CSPR。

啟動演示後，瀏覽器會開啟 `http://127.0.0.1:8765`。控制台可暫停、繼續、重播、修改 seed、調整皮帶/演示速度、調整中央協調器參數，並查看 A/B 任務、最新派工、拒絕原因與事件紀錄。關閉瀏覽器頁面不會停止模擬；在模擬仍開啟時，再次進入同一網址或雙擊 `open_dashboard.bat` 即可。MuJoCo 視窗聚焦後按 `R` 也會重播。

演算法的數學定義、程式對照與參數修改說明位於 [docs/algorithm_math_zh.md](docs/algorithm_math_zh.md)。
