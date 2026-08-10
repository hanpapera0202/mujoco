# 版本與推送規範

本專案使用語意化版本 `MAJOR.MINOR.PATCH`：

- `MAJOR`：改變控制架構、資料格式或場景契約，造成舊實驗結果不可直接比較。
- `MINOR`：新增可獨立使用的演算法、場景能力或控制台功能，且保留既有流程。
- `PATCH`：修正模型、參數、碰撞、視覺或文件，不改變公開契約。

每次推送須使用下列提交格式：

`vX.Y.Z: <scope> - <簡短英文摘要>`

其中 `<scope>` 可為 `sim`、`planner`、`grasp`、`model`、`dashboard`、`docs` 或 `build`。發布版本時必須建立並推送同名 Git tag，例如 `v0.2.0`。

目前版本 `0.13.0` 改用 `CR-RRIK`（Continuity-Regularized Resolved-Rate IK）。它以阻尼 Jacobian 求末端速度，加入零空間姿勢連續項與關節限位邊界，降低等價 IK 分支造成的手肘翻轉與腕部奇異姿勢。追蹤 IK 採 24 次 warm-start 迭代與 8 mm 位移門檻；安全待命與並行 handoff 規則維持不變。
