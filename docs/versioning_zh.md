# 版本與推送規範

本專案使用語意化版本 `MAJOR.MINOR.PATCH`：

- `MAJOR`：改變控制架構、資料格式或場景契約，造成舊實驗結果不可直接比較。
- `MINOR`：新增可獨立使用的演算法、場景能力或控制台功能，且保留既有流程。
- `PATCH`：修正模型、參數、碰撞、視覺或文件，不改變公開契約。

每次推送須使用下列提交格式：

`vX.Y.Z: <scope> - <簡短英文摘要>`

其中 `<scope>` 可為 `sim`、`planner`、`grasp`、`model`、`dashboard`、`docs` 或 `build`。發布版本時必須建立並推送同名 Git tag，例如 `v0.2.0`。

目前版本 `0.14.0` 延續 `CR-RRIK` 與雙臂 handoff 安全流程，新增獨立單臂場景。`nova5_single_arm_sorting_line.xml` 移除 B 臂、B 臂致動器/感測器與右側出料盤，但保留 8 節物理輸送帶、斜坡投料、動態物件、A 臂夾爪與左側出料盤；`run_single_arm_demo.py` 可直接啟動單臂 MuJoCo 演示。
