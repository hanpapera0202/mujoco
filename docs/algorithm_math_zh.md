# 中央協調演算法：數學、程式與修改指南

這份文件描述 **BC-JSP（Bayesian Centralized Joint Strategy Planner，貝式集中聯合策略規劃）**。CSPR 仍負責低成本快篩，BC-JSP 再處理兩臂聯合路徑。所有手臂在中央端平權，不存在主從關係。

## v0.6 不完全資訊貝式博弈

每臂提出三個動作 $R_a=\{direct,balanced,outer\}$，聯合策略集合為 $S=R_A\times R_B$，所以每輪固定只評估 9 組。MuJoCo 先在共同時間軸檢查實體碰撞與可調警戒盒；不安全策略直接令 $F_s=0$，權重不能覆蓋硬限制。

對尚未被有限取樣完全觀察的路徑安全事件，使用 Beta-Bernoulli 信念：

$$\theta_s\sim Beta(\alpha_s,\beta_s),\qquad E[\theta_s]=\frac{\alpha_s}{\alpha_s+\beta_s}.$$

成功證據令 $\alpha_s\leftarrow\alpha_s+1$，失敗、掉落或安全中止令 $\beta_s\leftarrow\beta_s+1$。策略完工機率為：

$$P_{finish}(s)=F_s E[\theta_s]q_Aq_B.$$

第一版的可解釋期望效用為：

$$U(s)=1000P_{finish}(s)-12T_{max}(s)+30\rho_{sim}(s)-0.25L(s)-400P_{collision}(s).$$

$T_{max}$ 是總完工時間，$\rho_{sim}$ 是兩臂同時有非零關節運動的取樣比例，$L$ 是兩臂總關節路徑長度。中央選擇 $s^*=\arg\max_{s\in S}U(s)$，因此可同時拒絕兩臂各自的局部最短路徑，避免局部貪婪造成類似布雷斯悖論的全局壅塞。若 9 組皆不安全，才退回單臂執行並保留另一任務。

## v0.8 密集式閉迴路 IK

MuJoCo 每 `2 ms` 取得物體狀態，關節位置控制維持 500 Hz。完整 360 次迭代 IK 不適合每個控制步重跑，因此抓取階段以先前關節解熱啟動，每 `40 ms` 執行最多 24 次阻尼 Jacobian 迭代，即 25 Hz。目標加入 `0.10 s` 前視：

$$p_g(t)=p_o(t)+0.10\dot p_o(t).$$

`prepare / track / descend / close` 均更新 $p_g(t)$；閉爪期間不再固定手腕位置。這些移動階段以時間與關節誤差判斷完成，不要求關節速度降為零，因為穩定跟帶本來就具有非零速度。

## v0.3 回授式執行修正

對手臂 $a$ 的每次完整任務量測 $T_a^{(k)}$，中央週期以指數移動平均更新：

$$\hat T_a \leftarrow 0.8\hat T_a + 0.2T_a^{(k)},\qquad \hat T=(\hat T_A+\hat T_B)/2.$$

投料必須滿足 $\lambda \leq \mu_A+\mu_B$，其中 $\lambda=1/\text{feed\_interval}$、$\mu_a=1/\hat T_a$。控制台因此提供投料間隔及最大線上物件數；超過負載的工件會留在上游。

候選任務建立關節時序預約 $\mathcal R_i=\{(t_j,q_i(t_j),g_i(t_j))\}$。中央端在同一 $t_j$ 同時套用候選與既有手臂姿態，僅於 $\forall t_j:C(q_A(t_j),q_B(t_j))=0$ 時放行。抓取硬條件為兩個指墊都接觸目標：$G_i=\mathbf1[F_{i,L}\cap O\ne\varnothing]\mathbf1[F_{i,R}\cap O\ne\varnothing]$。抓取 weld 已移除，放置只有通過輸出托盤空間驗證才計為成功。

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
