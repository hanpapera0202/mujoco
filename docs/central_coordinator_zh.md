# 中央協調器 v1

## 資料流

1. MuJoCo 提供物件位置、到達尾端的剩餘時間、類別與每隻手臂的抓取成功率。
2. 快篩逐一排除：非 `AVAILABLE`、超出 5 秒視窗、已承諾、手臂忙碌、超出可達範圍、無抓取候選、或無法在截止前完成的組合。
3. 對剩下的 arm-object pair 計算效益分數：期限急迫度、預測抓取成功率與移動距離。
4. 集中式匹配枚舉所有單臂與 A/B 配對，選出總效益最高的安全組合；雙臂同時作業有明確加分。
5. 選中的任務記錄為 `zone + enter_time + exit_time` 預約。預約是承諾，不在下一輪改派。
6. 低階 IK / 軌跡 / MuJoCo 執行器完成後回報 `PLACED` 或 `MISSED`。

## 工作區與路由

| 物件類別 | 可取件手臂 | 工作區 | 放置區 |
| --- | --- | --- | --- |
| `LEFT` | A | `exclusive_left` | `left_bin` |
| `MIDDLE` | A 或 B | `shared_middle` | A 取件則左側、B 取件則右側 |
| `RIGHT` | B | `exclusive_right` | `right_bin` |

共享區重疊的時間預約會被拒絕；不同專屬區或不重疊時間則可平行運行。這是第一版的安全抽象，之後可替換成連桿級距離預測與 5 cm 近失事件檢測。

## Benchmark

固定同一批種子、生成時間與低階執行器，至少跑 30 seeds。預設物件比例為 `LEFT 35% / MIDDLE 30% / RIGHT 35%`。結果輸出 JSONL 事件紀錄和 CSV 摘要，讓 Fuzzy、Deadline-first、Hungarian 與本中央時空預約方法可直接比較。
