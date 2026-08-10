# 版本與推送規範

本專案使用語意化版本 `MAJOR.MINOR.PATCH`：

- `MAJOR`：改變控制架構、資料格式或場景契約，造成舊實驗結果不可直接比較。
- `MINOR`：新增可獨立使用的演算法、場景能力或控制台功能，且保留既有流程。
- `PATCH`：修正模型、參數、碰撞、視覺或文件，不改變公開契約。

每次推送須使用下列提交格式：

`vX.Y.Z: <scope> - <簡短英文摘要>`

其中 `<scope>` 可為 `sim`、`planner`、`grasp`、`model`、`dashboard`、`docs` 或 `build`。發布版本時必須建立並推送同名 Git tag，例如 `v0.2.0`。

目前版本 `0.10.0` 延續 BC-JSP 與雙速率閉迴路 IK，新增 GUI 投料矩形範圍，以及中央交接預備狀態。當雙臂聯合路徑全部被 10 cm 警戒盒否決時，系統先將待命手臂送往 `handoff_escape`，再以 Bayesian readiness 控制接近 `handoff_ready` 的速度；領先手臂會在待命安全後啟動，完成後待命手臂立即接續抓取。預備狀態不會結算物件，也不會把物件誤記為 missed。
