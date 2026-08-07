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

`src/run_sorting_demo.py` 是建置展示用的少量投料場景。固定 seed 會產生 3 件 `MIDDLE / RIGHT / LEFT` 物件，中央協調器會先安排共享區與專屬區任務，再讓兩台 Nova5 以位置 IK 執行接近、下降、夾爪閉合、抬升、移至實體托盤、放開與回原位。附著只會在夾爪中心與物件通過實測對位門檻後建立，並保留抓取當刻的相對位移；這比直接傳送物件更接近物理抓取，但仍和後續純接觸摩擦抓取驗證分開。

啟動演示後，瀏覽器會開啟 `http://127.0.0.1:8765`。控制台可暫停、繼續、重播、修改 seed、調整皮帶/演示速度、調整中央協調器參數，並查看 A/B 任務、最新派工、拒絕原因與事件紀錄。關閉瀏覽器頁面不會停止模擬；在模擬仍開啟時，再次進入同一網址或雙擊 `open_dashboard.bat` 即可。MuJoCo 視窗聚焦後按 `R` 也會重播。

演算法的數學定義、程式對照與參數修改說明位於 [docs/algorithm_math_zh.md](docs/algorithm_math_zh.md)。
