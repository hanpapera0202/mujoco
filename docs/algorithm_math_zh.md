# 中央協調演算法：數學、程式與修改指南

這份文件描述第一版的高階協調器，命名為 **CSPR（Centralized Spatiotemporal Reservation，集中式時空預約）**。它不直接控制關節；MuJoCo 演示執行器在收到任務後，以位置 IK 完成取放。所有手臂在中央端是平等的，不存在主從關係。GUI 已預留 Deadline-first、Hungarian、Fuzzy 的切換欄位；目前只有 CSPR 已實作。

## 1. 集合與狀態

在時間 `t`，可見且尚未承諾的物件集合為 `O_t`，手臂集合為 `A={A,B}`。每件物件 `o` 有位置 `p_o`、到流水線尾端的剩餘期限 `D_o`、類別 `c_o in {LEFT,MIDDLE,RIGHT}`，以及每台手臂的預測抓取成功率 `q_{a,o}`。

每台手臂有目前工具位置 `p_a`、最大可達距離 `R_a`、忙碌結束時間 `b_a`。任務狀態為：

`AVAILABLE -> RESERVED -> PICKED -> PLACED`，或在尾端離開時變為 `MISSED`。

## 2. 快篩

中央端先對每個 arm-object pair 建立硬式可行性指標：

```text
F(a,o) = I[AVAILABLE] I[D_o <= H] I[t >= b_a] I[||p_a-p_o|| <= R_a]
         I[eta(a,o) < D_o] I[q_(a,o) > 0] I[class_allowed(a,o)]
```

其中：

```text
d(a,o)   = ||p_a - p_o||
eta(a,o) = d(a,o) / v_pick + T_cycle
```

`H` 是滾動視窗；`v_pick` 是高階估計的取件移動速度；`T_cycle` 是抓取、放置等固定工作時間。`class_allowed` 使 `LEFT` 只由 A、`RIGHT` 只由 B 處理，而 `MIDDLE` 可由任一手臂取件。

程式位置：[central_coordinator.py](../src/central_coordinator.py) 的 `_screen_pair()`。

## 3. 候選效益

通過快篩的候選任務 `(a,o)` 得分為：

```text
u(o) = 1 / max(D_o, epsilon)
s(a,o) = w_u u(o) + w_q q_(a,o) - w_d d(a,o)
```

預設值為 `w_u=3.0`、`w_q=2.0`、`w_d=0.25`。因此期限更近、抓取成功率更高、距離更短的候選分數較高。這些值不是安全限制，而是可比較的偏好；GUI 的 Coordinator 區可調整，按 `Apply And Restart` 後套用。

## 4. 集中式配對與並行偏好

令二元變數 `x_(a,o)=1` 表示將物件 `o` 指給手臂 `a`。第一版只有兩台手臂，所以程式精確枚舉所有單一候選與 A/B 候選對，而不需要近似求解器：

```text
max_x  sum_(a,o) x_(a,o) s(a,o) + beta I[both A and B assigned]
```

約束如下：

```text
sum_o x_(a,o) <= 1             for each arm a
sum_a x_(a,o) <= 1             for each object o
x_(a,o) = 0                    if F(a,o)=0
```

`beta` 是 `parallel_bonus`。它使兩台手臂各自有一項安全任務時，系統偏好同時工作，即使其中一個單項分數略低。程式位置是 `_choose_global_assignment()`。

## 5. 共享區時空預約

每個候選會提出時間區間：

```text
I_(a,o) = [t + d(a,o)/v_pick, t + eta(a,o)]
```

對共享區 `shared_middle`，兩項任務不能同時預約重疊區間：

```text
not overlap(I_i, I_j)
overlap([l1,r1], [l2,r2]) iff max(l1,l2) < min(r1,r2)
```

因此 `MIDDLE` 物件形成真正的協作壓力；如果一隻手臂獲得共享區任務，另一隻手臂會在同一輪轉向自己的 `LEFT` 或 `RIGHT` 專屬件。預約建立後不重新指派，降低集中協調器的計算與狀態複雜度。

MuJoCo 執行層再加入中央走廊門檻。若另一手臂正處於 `approach / descend / close / lift`，共享件維持 `RESERVED` 但不進入軌跡；待對方進入 `to_bin` 後才啟動。這是以執行階段補足高階 zone-time 預約的保守安全條件。每一步還會檢查 A/B 幾何接觸；任何跨手臂接觸都會觸發安全暫停。

## 6. 可調參數

| GUI 欄位 | 符號 | 影響 | 建議起點 |
| --- | --- | --- | --- |
| Rolling horizon | `H` | 可見的未來任務範圍 | 5.0 s |
| Parallel bonus | `beta` | 雙臂同時動作偏好 | 2.0 |
| Pick speed | `v_pick` | 截止前可行性的預估 | 0.55 m/s |
| Fixed cycle | `T_cycle` | 抓取/放置固定時間 | 1.1 s |
| Urgency weight | `w_u` | 期限優先程度 | 3.0 |
| Success weight | `w_q` | 成功率優先程度 | 2.0 |
| Travel weight | `w_d` | 移動距離懲罰 | 0.25 |
| Belt speed | `v_belt` | 物理皮帶與物件截止壓力 | 0.18 m/s |

修改時建議一次只改一個參數並保留 seed。較大的 `beta` 會提高平行率；較大的 `w_u` 會更積極搶救接近尾端的物件；過高的 `v_belt` 可能使固定 `T_cycle` 下的任務不再可行。安全相關的可達性與共享區重疊仍是硬限制，不會被權重覆蓋。

## 7. 與 MuJoCo 的界線

演示器的 IK、路徑碰撞預檢與接觸觸發夾持位於 [run_sorting_demo.py](../src/run_sorting_demo.py)。它以固定 seed 驗證中央端的任務序列、共享區保留與輸出托盤路由。工件不使用座標式運動學附著；只有 MuJoCo 接觸成立才會啟用夾持約束，`q_(a,o)`、`eta(a,o)` 和近失事件可直接由接觸與落盤驗證結果提供。
