# 論文低階架構 Standard

## Standard 身分

目前先建立 `paper-low-level-seed42` 作為**可重現基線**，不是宣稱通過的最終
性能標準。完整參數在 `configs/standard_low_level_benchmark.yaml`，固定：

- `seed=42`、模擬窗口 60 秒。
- 皮帶 `0.09 m/s`，每 5 秒投 1 件，總計 10 件，最大線上物件 10 件。
- MuJoCo `0.002 s` timestep，500 Hz 控制、25 Hz tracking IK、50 Hz 安全預測。
- 碰撞警戒盒外擴 10 cm。
- 高階 `BC-GP-JSP + CSPR`，低階 `paper_closed_chain_low_level`，執行 `QP-RRIK`。
- 一般輸送帶任務是 `independent_pick`；只有共同持物/交接才切換
  `closed_chain`。

## 實時基線結果

已開啟最新 MuJoCo 與 Web GUI，固定 seed 42 觀察到模擬時間 `38.94 s`：

| 指標 | 結果 | 判讀 |
|---|---:|---|
| 已投料 | 8 | 產線持續投料 |
| 已放置 | 2 | 目前不足 |
| 漏件 | 1 | 不符合最終目標 |
| 線上物件 | 5 | 目前出現服務率壓力 |
| A/B 抓取成功 | 1 / 1 | 雙指實體抓取路徑有工作 |
| swept path reject | 3 | 低階路徑/碰撞預檢仍過度拒絕 |
| path abort | 1 | B 指墊與 floor 禁止接觸，不能接受 |
| safety stop | 0 | 尚未觸發整體安全停機 |

低階 GUI 顯示已是「論文閉鏈低階層」，目前模式是獨立物件分揀；這是正確的，
因為本次兩臂沒有共同夾同一個剛體。

## Standard 的用途

每次更換 IK、閉鏈約束、皮帶速度或碰撞盒後，先用這組固定參數重播，再比較：

1. 是否仍維持 500/25/50 Hz 與無 NaN/Inf。
2. 是否零 arm-arm、arm-body、arm-floor 禁止接觸。
3. 是否雙指接觸後才記錄 grasp，並只在目標托盤內 release。
4. 60 秒內放置率和漏件數是否比這次基線改善。
5. `swept_path_reject` 是否下降，而不是只靠等待把問題藏起來。

目前基線暴露的主要卡點是 B 進入抓取/回復路徑時的地板穿越風險，以及 10 cm
警戒盒造成的共享走廊等待。下一步應先修低階的安全姿態生成與路徑重規劃，之後
再調高階派工權重；否則只改投料間隔會掩蓋根因。
