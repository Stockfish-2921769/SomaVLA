# 规划场可视化资产

对应 `EXPERIMENT_LOG.md` 轴 F（2026-10-17）与事后审计（2026-10-18）。

| 文件 | 内容 |
|---|---|
| `scene.png` | 多物体场景渲染（PIL 顶视 256×256），含目标、已放置物体、双墙 |
| `fields.png` | 障碍栅格 / GT 测地场 / NCA 预测场，叠加起点、目标与 escape 路径 |
| `ordering.png` | 沿 GT 下降路径的 GT 距离与 NCA 场值；红叉为符号违例步 |
| `executor.gif` | 阻尼 plant 跟随 NCA escape 路径的执行动画 |

重新生成：

```bash
PLAN_CKPT=/path/to/plan_nca_long3r.pt python scripts/audit/make_viz.py
```

注意：可视化使用的 27.5K NCA checkpoint 属于训练产物（`*.pt` 已被 `.gitignore` 忽略），
仓库不携带权重；请用 `scripts/train_multi_plan.py` 复现，或让 `PLAN_CKPT` 指向本地副本。
