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

# 執行可重現的第一版基準範例
python src\run_benchmark.py --seeds 30 --output-dir results\v1
```

基準輸出包含 `events.jsonl`（每次決策與結果）與 `metrics.csv`（漏件率、正確分流率、平均取件時間、近失次數、雙臂同時工作比例）。目前基準使用固定時間模型；下一階段會讓 `run_sorting_line.py` 回傳 MuJoCo 實測事件。
