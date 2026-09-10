# Nova5 專案記憶與驗收契約

這份文件是需求的唯一長期台帳。每次新增要求時，先新增一個需求 ID、預期行為與驗證方法，再修改程式。`PASS` 代表後續版本不可破壞；`OPEN` 代表已知尚未完成，不能假裝成功。

## 目前不可退化的需求

| ID | 狀態 | 需求 | 自動驗證 |
|---|---|---|---|
| ARCH-01 | PASS | 兩台 Nova5 由同一個集中式 BC-JSP 協調，沒有主從手臂；CSPR 作為快速硬條件篩選層 | `test_central_coordinator.py`、`test_bayesian_joint_planner.py` |
| PAR-01 | ARCHIVED v0.4 | 固定 seed 42 的第一對左右物件由 A、B 同時執行 | Git tag `v0.4.0` |
| GRASP-01 | PASS | 抓取必須同時接觸左右兩個實體指墊 | `test_sorting_demo_regression.py` |
| GRASP-02 | PASS | 抓取不得啟用 weld/equality constraint；雙指實體接觸後可用已驗證的 transport attachment 維持演示統計，放置時解除 | `test_sorting_demo_regression.py` |
| SAFE-01 | PASS | 固定 seed 物理抓放期間不得發生雙臂碰撞或安全停止 | `test_sorting_demo_regression.py` |
| LOAD-01 | PASS | 投料間隔及最大線上物件數可設定，避免輸入率超過服務率 | 程式參數與控制台 |
| PLACE-00 | PASS | 固定 seed 42 的首件須維持雙指夾持至釋放，並落入中央分配的正確托盤 | `test_sorting_demo_regression.py` |
| PLACE-01 | PASS | 演示模式以已驗證雙指接觸後的 transport attachment 維持物件，seed 42 十件皆放入正確托盤 | `standard_low_level_benchmark.yaml`、90 秒 headless 驗證 |
| GUI-01 | PASS | 中文控制台可開啟或重新開啟對應的 MuJoCo 視窗；開始動作會同時開啟視窗 | `test_demo_dashboard.py` |
| PROFILE-01 | PASS | GUI 可用 ASCII key 儲存與載入完整 seed/參數設定；`nova5_fb1s_fast10_v038` 可重現 seed 42 的快速十件基準 | `test_demo_dashboard.py`、`configs/reproducible_profiles/` |
| LINE-01 | PASS | v0.5 產線有效皮帶寬度為 45 cm，兩台 Nova5 基座位於 x = +/-0.38 m、y = 0.15 m | `test_narrow_middle_scenario.py` |
| ALLOC-01 | PASS | v0.5 的 10 件皆為 MIDDLE；完全同分首件 A 優先，後續按負載維持平行權限 | `test_central_coordinator.py`、`test_narrow_middle_scenario.py` |
| SAFE-02 | PASS | 每臂以上臂、前臂、夾爪三個長方體作粗略模型；預警外擴可由 GUI 調整且不改實體盒，預設 10 cm | `test_narrow_middle_scenario.py` |
| PAR-02 | OPEN | 全 MIDDLE 動態產線提高雙臂同時運動比例；目前 10 cm 領域會讓部分中央預約等待 | 待新增重疊時間指標與協調路徑 |
| JOINT-01 | PASS | 中央處理器比較 A/B 各三條路徑的 9 組聯合策略，不固定讓路手臂；可拒絕兩條局部最短路徑 | `test_bayesian_joint_planner.py` |
| JOINT-02 | OPEN | 可行聯合方案中兩臂須同時運動；只允許減速，除非所有聯合候選皆不安全才可停止 | 待新增雙臂速度重疊指標 |
| JOINT-03 | PASS | 聯合路徑先通過三盒碰撞硬限制，再依貝式完工機率、總工期、同動比例與路徑長度計算效用 | `test_bayesian_joint_planner.py`、`test_narrow_middle_scenario.py` |
| BAYES-01 | PASS | 路徑安全不確定性以 Beta-Bernoulli 信念表示，成功或失敗證據可更新後驗 | `test_bayesian_joint_planner.py` |
| DYN-02 | PASS | 安全減速後按剩餘閉爪時間更新動態攔截點；A 與 B 均須能對移動 MIDDLE 物件形成雙指接觸 | `test_dynamic_intercept.py`、`test_sorting_demo_regression.py` |
| FEED-02 | PASS | GUI 可設定每批投料物品數，現行預設一次投放 1 件，投料間隔套用於批次之間 | `test_narrow_middle_scenario.py` |
| PREP-01 | PASS | 抓取任務先執行提前準備與跟帶靠近，再下降閉爪，不得只在最終位置突然啟動 | `test_dynamic_intercept.py`、任務階段快照 |
| FEED-03 | PASS | GUI 可調投料 X/Y 最小與最大值；固定 seed 可重現，預設批次左右分散且不固定中線 | `test_narrow_middle_scenario.py` |
| TRACK-01 | PASS | 物體狀態 500 Hz 讀取、熱啟動 IK 25 Hz；閉爪期間仍跟隨皮帶，移動階段不得要求關節速度歸零 | `test_dynamic_intercept.py`、`test_sorting_demo_regression.py` |
| CCK-01 | PASS | 以論文方法提供三點基座校正、共同工件閉鏈位姿殘差與平滑時間尺度；一般輸送帶分揀不強制共同閉鏈 | `test_closed_chain_kinematics.py` |
| TRAJ-01 | PASS | 聯合候選加入速度/加速度平滑度軟成本；碰撞、關節限位與接觸仍是硬條件 | `test_closed_chain_kinematics.py`、`test_bayesian_joint_planner.py` |
| TIMING-01 | OPEN | 可量測一臂離開皮帶到 peer 進入皮帶上方的延遲；實驗分支預設期限 0.80 s 並可強制加速 | 待完成固定 seed 10 件物理驗證 |
| SIM-01 | OPEN | 最低運行優先模式取消 A/B 粗略碰撞盒彼此接觸，並將指墊擦地列為非致命；手臂本體對皮帶、地板與機台仍防護，可切回 strict | 待完成雙臂並行運行與碰撞紀錄驗證 |
| ZONE-01 | PASS | 共享皮帶切成前/後兩段（分界 Y=0.25 m）；各段紅燈代表占用、黃燈代表離開後 1 秒冷卻、綠燈代表可讓另一臂進入；不同段可並行且同區進入具原子預約 | `test_narrow_middle_scenario.py`、GUI/API、seed 42 十件動態驗證 |

## 新需求加入方式

例如要加入「不同皮帶速度仍可抓取」，先在上表增加：

```text
DYN-01 | OPEN | 皮帶 0.18、0.24、0.30 m/s 都能雙指抓取 | 待新增參數化測試
```

接著才修改模型或演算法。完成後新增測試並把狀態改成 `PASS`。若舊的 `PASS` 測試失敗，代表新功能造成退化，先修正退化，不可刪除測試。

## 每次修改後的檢查

在 PyCharm Terminal 或 PowerShell 執行：

```powershell
cd C:\Users\Han\PycharmProjects\mujoco
python -m unittest discover -s tests -v
```

看到所有測試都是 `OK` 才能建立版本。GitHub 的 `.github/workflows/regression.yml` 也會在每次 push 和 pull request 自動執行相同檢查；綠色勾表示既有能力沒有被破壞，紅色叉表示要先修復。
