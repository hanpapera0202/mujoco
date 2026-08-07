# Nova5 Dual-Arm Sorting Research

此 repo 是 Nova5 雙手臂皮帶分揀的 MuJoCo 研究平台。研究重點是集中式、Deadline-aware 的雙手臂任務分配與共享工作區時空預約。

## 快速開始

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src\run_sorting_line.py
```

## 專案結構

```text
models/nova5/       MuJoCo 產線場景
models/meshes/      Nova5 STL meshes
configs/            皮帶與投料設定
src/run_sorting_line.py
                    物理皮帶與隨機斜坡投料控制器
src/central_coordinator.py
                    集中式快篩、全域派工與時空 reservation
docs/               演算法設計文件
```

## 目前架構

協調器先排除 Deadline 已失效、不可達、忙碌或無抓取候選的 arm-object pair；其後以抓取成功率、Deadline 與移動成本排序，並為完整共享碰撞空間建立 `zone + time interval` reservation。A、B 可同時動作，只有時空區間衝突時才讓其中一方等待或重規劃。
