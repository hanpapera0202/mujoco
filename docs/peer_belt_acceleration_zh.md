# Peer 離帶後加速分支

分支：`experiment/peer-belt-acceleration`

## 時序定義

- `belt_departure`：一臂完成 `lift`、進入 `to_bin`。
- `peer_belt_entry`：另一臂首次進入 `track`、`descend`、`close` 或 `lift`。
- `latency_s = t_entry - t_departure`。

預設期限為 `0.80 s`。期限到而 peer 尚未進入時，中央執行器會將等待任務提升成
正式任務，並將尚未完成階段乘以 `peer_motion_speed_scale=0.60`。硬碰撞、地板
接觸、關節限制和雙指抓取條件不會因加速而取消。

## 低效能觀測

MuJoCo 每 `0.50 s` 記錄一次兩臂 stage、物件、抓取點位置與關節運動量；GUI 的
「離帶後 Peer 時序」顯示期限、最近 latency、超時及強制提升次數。這比每個 2 ms
截圖便宜很多，也足以辨認「另一臂在等待」是派工、預檢還是控制器問題。

## 新的 10 件搜尋門檻

搜尋時固定 `seed`、`feed_batch_size=1`、`max_active_parts=10`，只接受：

```text
placed == 10 and missed == 0 and paused == false
```

先由寬間隔找出第一個通過點，再逐步縮短 `feed_interval_s`。若所有間隔都因同一
個 floor/arm collision 失敗，停止縮短並先修低階姿態；那不是產線間隔的性能
平台，而是控制器安全錯誤。
