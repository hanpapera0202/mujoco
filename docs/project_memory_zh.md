# Nova5 專案記憶與驗收契約

這份文件是需求的唯一長期台帳。每次新增要求時，先新增一個需求 ID、預期行為與驗證方法，再修改程式。`PASS` 代表後續版本不可破壞；`OPEN` 代表已知尚未完成，不能假裝成功。

## 目前不可退化的需求

| ID | 狀態 | 需求 | 自動驗證 |
|---|---|---|---|
| ARCH-01 | PASS | 兩台 Nova5 由同一個集中式 CSPR 協調，沒有主從手臂 | `test_central_coordinator.py` |
| PAR-01 | PASS | 固定 seed 42 的第一對左右物件由 A、B 同時執行 | `test_sorting_demo_regression.py` |
| GRASP-01 | PASS | 抓取必須同時接觸左右兩個實體指墊 | `test_sorting_demo_regression.py` |
| GRASP-02 | PASS | 抓取不得啟用 weld/equality constraint，也不得瞬移物件 | `test_sorting_demo_regression.py` |
| SAFE-01 | PASS | 第一對並行抓取期間不得發生雙臂碰撞或安全停止 | `test_sorting_demo_regression.py` |
| LOAD-01 | PASS | 投料間隔及最大線上物件數可設定，避免輸入率超過服務率 | 程式參數與控制台 |
| PLACE-01 | OPEN | 物件須靠真實夾持力運送並落入正確托盤，10 件完成率目標 100% | 尚待低階動力控制完成 |

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

