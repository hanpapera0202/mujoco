# Nova5 多機械臂時空 Voxel 運動規劃

## 1. 研究目標

本專案先在 MuJoCo 數位孿生環境中，以兩支 DOBOT Nova5 建立可擴展至多機械臂的時空碰撞篩選式運動規劃原型。

核心流程：

```text
Task
→ Multiple Motion Candidates
→ Kinematic Filtering
→ Trajectory Generation
→ Future Spatiotemporal Occupancy
→ Collision Filtering
→ Best Safe Motion
```

研究重點不是只判斷「目前是否碰撞」，而是提前描述每支機械臂在未來各時間點會占用哪些三維空間，並利用這些資訊篩除彼此在時空上衝突的候選運動。

## 2. MuJoCo 與 Shared Workspace

- 模擬平台：MuJoCo
- 機械臂：雙 DOBOT Nova5
- 任務：先以 Pick-and-Place 為主要測試任務
- Shared Workspace：兩支機械臂可能互相干涉的中央共享工作空間
- Shared Workspace 劃分為 `5 × 5 × 5 = 125` 個固定 voxel

### Voxel

Voxel 是固定於三維空間中的小立方格，用來表示空間位置與占用狀態。

每個 voxel 需要唯一 `voxel_id`。

第一版 occupancy 資料至少描述：

```text
(voxel_id, robot_id, time, occupied)
```

其中 `robot_id` 用來區分不同機械臂，例如 `left_arm` 與 `right_arm`。

## 3. Occupancy 與 Spatiotemporal Map

### Occupancy

Occupancy 表示某一時間點，某支機械臂的實體幾何是否占用指定 voxel。

### Temporal Occupancy

Temporal Occupancy 進一步加入時間資訊：

```text
time t0 → robot A occupies {v3, v4, v8, ...}
time t1 → robot A occupies {v4, v5, v9, ...}
time t2 → robot A occupies {v5, v6, v10, ...}
```

### Spatiotemporal Map

Spatiotemporal Map 是由三維空間 `(x, y, z)` 與時間 `t` 組成的未來占用表示。

重要原則：

> 只有「相同空間」不足以構成衝突；必須同時考慮「相同時間」。

因此兩條幾何路徑可以經過相同 voxel，只要兩支機械臂到達該 voxel 的時間沒有重疊，就不應被判定為時空衝突。

## 4. Voxel Visualization

MuJoCo viewer 中以半透明 voxel 顯示 Shared Workspace，使機械臂與 voxel 可以同時觀察。

第一版顏色定義：

- 紅色：目前或指定時間內被機械臂占用
- 黃色：狀態轉換的視覺提示，例如剛離開或接近釋放
- 綠色：目前可使用

紅、黃、綠主要是可視化介面；後續規劃演算法真正使用的是 future occupancy data，而不是只依靠顏色判斷。

## 5. Pick-and-Place Motion Candidates

同一個抓取／放置任務不只保留單一運動解，而是產生多個 Motion Candidate。

### Grasp Pose

Grasp Pose 是夾爪實際抓取物體時的末端位姿，包含：

- Position：末端位置
- Orientation：末端旋轉方向

### Approach Pose

Approach Pose 是正式進入 Grasp Pose 前的預抓取位姿，用來決定機械臂從哪個方向接近物體。

### IK — Inverse Kinematics

Inverse Kinematics（逆運動學）是在已知末端目標 Pose 時，求得對應的關節角：

```text
Target end-effector pose → q = [q1, q2, ..., q6]
```

相同末端 Pose 可能存在多組 IK Solution。有效的不同 IK branch 應保留成不同候選構型，而不是只保留距離目前關節角最近的一組。

Motion Candidate 主要由以下差異產生：

1. 不同 Grasp / End-effector Orientation
2. 不同 Approach Pose / Approach Direction
3. 不同有效 IK Branch / Configuration
4. Joint-space 或 Cartesian-space 運動方式

候選方案應來自具有運動學意義的不同構型，而不是單純使用大量隨機 waypoint 製造數量。

## 6. Kinematic Filtering

Motion Candidate 在進入時空碰撞篩選前，先排除自身不可行方案。

第一版至少包含：

- Joint Limit：關節角不可超過允許範圍
- Self-Collision：機械臂本體不可自碰撞
- Static Collision：不可與固定環境或障礙物碰撞
- Singularity：避免接近嚴重奇異姿態

## 7. Singularity Detection

### Jacobian

Jacobian `J(q)` 描述關節速度與末端速度之間的關係。

### Minimum Singular Value

第一版主要使用 Jacobian 的最小奇異值：

```text
sigma_min(J)
```

作為奇異性指標。

當 `sigma_min(J)` 接近 0，表示機械臂接近奇異構型。

奇異性檢查不能只檢查最終 Goal Configuration，必須沿完整 trajectory 取樣：

```text
q(t0), q(t1), ..., q(tN)
```

如果 trajectory 中存在嚴重奇異區域，該 Motion Candidate 應被淘汰。

## 8. Trajectory

Trajectory 表示完整的關節時間函數 `q(t)`，不只是起點與終點。

第一版優先使用容易驗證的方法：

- Joint-space interpolation：直接在關節空間插值
- Cartesian interpolation：先指定末端 Cartesian path，再轉換為對應的 joint trajectory

第一階段不需要先導入複雜的全域最佳化 planner。

## 9. Trajectory Sampling 與 Future Occupancy

每條 candidate trajectory 使用固定時間步長 `Δt` 取樣：

```text
t0, t1, t2, ..., tN
```

每個 sample 執行：

```text
q(t)
→ robot geometry in MuJoCo
→ occupied voxel IDs
```

形成：

```text
time → robot_id → occupied voxel IDs
```

Future Occupancy 只需要保存有限的 Prediction Horizon `Th`，即系統一次向未來預測的時間範圍。

## 10. Spatiotemporal Conflict

第一版採保守且明確的 collision rule：

> 若兩支機械臂在同一 time slice 占用同一 voxel，則該 candidate trajectory 存在 Spatiotemporal Conflict。

可表示為：

```text
same voxel + overlapping time → conflict
```

發生衝突的 candidate 直接淘汰。

後續可在此基礎上加入 spatial safety margin 與 temporal safety margin。

## 11. Safe Candidate Ranking

通過自身可行性與 Spatiotemporal Conflict Filtering 後，剩餘方案形成 Safe Candidate Set。

第一版使用以下指標排序：

- Execution Time：完成 trajectory 所需時間
- Total Joint Motion：各關節在整條 trajectory 中的累積移動量
- Smoothness：速度／加速度變化是否平順

最後選出 Best Safe Motion 執行。

## 12. 開發順序

### Phase 0 — 125 Voxel Visualization

建立 `5 × 5 × 5 = 125` voxel Shared Workspace，確認：

- 半透明顯示
- Nova5 與 voxel 同時可見
- 占用顏色更新正確

### Phase 1 — Occupancy Data Interface

將 occupancy detection 與 visualization 分離，提供程式可直接取得：

```text
time → robot_id → occupied voxel IDs
```

### Phase 2 — Motion Candidate Generation

建立 Pick target / Grasp target，生成多種：

- Grasp Orientation
- Approach Pose
- IK Solution

### Phase 3 — Kinematic Filtering

加入：

- Joint Limit
- Self-Collision
- Static Collision
- Trajectory Singularity Check

### Phase 4 — Trajectory Generation

由可行構型產生多條 Joint-space / Cartesian-space trajectory。

### Phase 5 — Future Spatiotemporal Occupancy

將各 trajectory 轉換為：

```text
(x, y, z, t, robot_id)
```

或等價 occupancy representation。

### Phase 6 — Spatiotemporal Collision Filtering

比較 candidate 與其他機械臂 future occupancy，淘汰時空衝突方案。

### Phase 7 — Best Safe Motion Selection

依 Execution Time、Total Joint Motion 與 Smoothness 選出最終執行方案。
