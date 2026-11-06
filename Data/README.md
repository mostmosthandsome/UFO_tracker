# DR02 LaFAN 训练数据

来源：`/home/handsome/repo/handsome/research/DataSet/LAFAN1/dr02_csv` 中全部 77 个 CSV。
转换日期：2026-11-05。此数据集仅包含该目录的动作，没有额外混入此前的两条侧手翻数据。

| 文件 | 用途 |
|---|---|
| `dr02_lafan_10s_clipped.pkl` | 训练输入，1,692 个片段 |
| `dr02_lafan_full.pkl` | 完整参考动作，77 条，供跟踪推理使用 |
| `dr02_lafan.yaml` | 训练/推理数据清单，关联 DR02 机器人配置 |
| `conversion_report.json` | 每个源文件的哈希、帧数、时间检查、关节顺序和限位检查 |
| `validation_report.json` | 保存后重新读取并与所有 CSV/完整动作逐项对照的结果 |
| `convert_dr02_lafan.py` | 可复现转换脚本，调用仓库原生 RobotState → UFO 转换器 |

共 496,595 帧，30 Hz，约 4.598 小时。按 10 秒（300 帧）连续切片，保留至少 1 秒的尾段；
本次所有尾段均满足条件，训练数据覆盖所有源帧，无丢弃和重复。
没有重采样、平移根位置、调整地面高度、减去默认关节姿态或裁剪关节角。

CSV 实际表头为 `time,root_x,root_y,root_z,root_qx,root_qy,root_qz,root_qw,dof_<joint_name>...`。
23 个关节列采用腰、左臂、右臂、左腿、右腿顺序；以表头名称映射到
`configs/robots/dr02_23dof.yaml` 的 `control_joints`，不套用腿优先列表。
读取时跳过 `#` 注释并保留其元信息；以 `sample_rate: 30` 为采样率，逐文件检查时间列一致性。
根四元数为 XYZW，归一化后存储。坐标按源数据的 Z-up、米和弧度解释。
未发现非有限数值、无效四元数或超出模型关节限位超过 1e-4 rad 的角度。

PKL 用 `joblib.load` 读取，结构为 `{动作名称: 动作记录}`。主要字段：

- `root_trans_offset: (T,3)`：根部世界位置。
- `root_quat: (T,4)`：根部 XYZW 四元数。
- `dof_pos: (T,23)`：机器人受控关节角。
- `pose_aa: (T,32,3)`：按 DR02 body 顺序排列的轴角，包含根节点和固定节点。
- `fps: 30.0`、`joint_names`、`body_names`、来源和切片元信息。

速度及 FK 等派生量由 UFO MotionLib 在加载时计算。完整动作与训练切片的 motion ID 不同。
数据格式和内容一致性验证不等于策略训练收敛或动作动态可执行性验证。

## 启动训练

在 UFO 仓库根目录、已安装仓库训练依赖的环境中运行：

```bash
./run_train.sh --agent fb \
  --robot-config configs/robots/dr02_23dof.yaml \
  --data-manifest Data/dr02_lafan.yaml \
  --gpu-ids single \
  --work-dir runs/dr02_lafan
```

这次转换没有启动训练。原 `configs/data/dr02_local.yaml` 仍指向此前两条侧手翻，
若要使用本批数据，必须采用上面的新清单，或显式传入
`--data-path Data/dr02_lafan_10s_clipped.pkl`。

## 重新转换

在仓库根目录运行（需要 numpy、mujoco、joblib、loguru、omegaconf）：

```bash
python Data/convert_dr02_lafan.py \
  --source /home/handsome/repo/handsome/research/DataSet/LAFAN1/dr02_csv \
  --output /tmp/dr02_lafan_rebuilt
```

脚本拒绝覆盖已有输出文件。`dr02_lafan.yaml` 中的数据路径相对于清单，机器人配置路径为
本机绝对路径；复制到其他机器后需调整 `robot_config`。
