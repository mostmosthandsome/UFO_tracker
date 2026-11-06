# UFO 纯网络推理：文件、输入和编码说明

本目录仅依赖 pip 可安装的 `torch`、`safetensors`、`numpy`、`scipy`，使用 Python 3.10+。
不导入 UFO、Gymnasium、MuJoCo、mjlab 或 Isaac Lab，不执行仿真。

`infer.py` 提供两个接口：

```python
z = policy.encode(ref_action)  # 参考状态特征 → 任务潜变量
action = policy(obs, z)        # 机器人当前观测 + 任务潜变量 → 动作
```

`encode` 对 FB 使用 backward map（B），对 TeCH 使用 goal encoder。
Actor 和编码器共享 checkpoint 保存的观测归一化统计，但加载不同的网络权重。
编码器仅在首次调用 `encode` 时加载；不会加载 critic、优化器或 replay buffer。

**`encode` 支持 G1 的 `(L,29)` 关节角数组，须同时传入根位置、根姿态和 FPS。**
内部调用 `UFO_observation.py` 计算参考观测，再送入 B。原有特征字典输入仍然支持。

```python
import torch
from infer import Policy

policy = Policy('/path/to/checkpoint')
L = 100
q = torch.zeros(L, 29)                 # 替换为参考关节角，rad
root_pos = torch.zeros(L, 3)
root_pos[:, 2] = 0.8                  # 演示根高度；真实动作需提供实际轨迹
root_quat = torch.tensor([0., 0., 0., 1.]).repeat(L, 1)  # XYZW 单位四元数
z = policy.encode(q, root_pos=root_pos, root_quat=root_quat, fps=30)
```

默认使用随目录附带的 G1 XML 和 `UFO_observation_compute/g1_metadata.json`，
包含 29 个关节顺序、默认姿态和虚拟 head_link；默认启用根高度、观测尺度均为 1。
输入列顺序不同时传 `joint_names=[...]`。实际 checkpoint 若使用其他训练参数，
用 `Policy(..., reference_metadata=metadata_dict, reference_xml='/path/to/robot.xml')` 覆盖。
网络配置不能还原所有机器人观测参数，维度一致也不代表尺度/顺序一致。
可选 `sample_times` 和 `drop_first=True` 用于复现 tracking 的控制采样及下一帧对齐。

### G1 默认 joint_names 顺序

`encode` 的参数默认是 `joint_names=None`，表示输入 `ref_action[:, i]` 已按下面的
`control_joint_names` 顺序排列。此列表来自附带的 `UFO_observation_compute/g1_metadata.json`，
注释为从 0 开始的列索引：

```python
joint_names = [
    "left_hip_pitch_joint",       # 0
    "left_hip_roll_joint",        # 1
    "left_hip_yaw_joint",         # 2
    "left_knee_joint",            # 3
    "left_ankle_pitch_joint",     # 4
    "left_ankle_roll_joint",      # 5
    "right_hip_pitch_joint",      # 6
    "right_hip_roll_joint",       # 7
    "right_hip_yaw_joint",        # 8
    "right_knee_joint",           # 9
    "right_ankle_pitch_joint",    # 10
    "right_ankle_roll_joint",     # 11
    "waist_yaw_joint",            # 12
    "waist_roll_joint",           # 13
    "waist_pitch_joint",          # 14
    "left_shoulder_pitch_joint",  # 15
    "left_shoulder_roll_joint",   # 16
    "left_shoulder_yaw_joint",    # 17
    "left_elbow_joint",           # 18
    "left_wrist_roll_joint",      # 19
    "left_wrist_pitch_joint",     # 20
    "left_wrist_yaw_joint",       # 21
    "right_shoulder_pitch_joint", # 22
    "right_shoulder_roll_joint",  # 23
    "right_shoulder_yaw_joint",   # 24
    "right_elbow_joint",          # 25
    "right_wrist_roll_joint",     # 26
    "right_wrist_pitch_joint",    # 27
    "right_wrist_yaw_joint",      # 28
]
```

若动作文件采用其他列顺序，应将其实际列名作为 `joint_names` 传入，程序会按名称重排。
若通过 `reference_metadata` 覆盖配置，省略 `joint_names` 时使用的是新配置中的
`control_joint_names` 顺序。

**准备参考动作时，最小原始数据是：关节角序列 + 根部位置序列 + 根部朝向序列 + FPS。**
另需一次性准备匹配的模型权重、机器人模型和训练配置。
各 body 位姿、速度、投影重力、`state`、`privileged_state` 和 z 都可以计算，
不必作为独立的原始数据额外采集。具体分类见第 4 节。

## 1. 必需文件

纯网络加载仅需要 `checkpoint/model/` 内的三个文件：

```text
model/
├── config.json         # Actor/B 架构、输入字段顺序、z 维度、归一化配置
├── init_kwargs.json    # 观测字段维度、动作维度
└── model.safetensors   # 网络权重以及观测归一化统计量
```

网络加载不需要实验根目录的 `config.json`、`config.yaml`，也不需要
`checkpoint/config.json`、`optimizers.pth`、`train_status.json` 或 `buffers/`。

但如果要从原始关节角/传感器数据构造真实输入，需要额外查阅训练时的配置和机器人模型，
以获得关节/body 顺序、默认姿态、观测缩放、历史顺序、控制周期和动作处理规则。
这些信息不能仅由上述三个文件中的网络维度完整还原。
可参考实验根目录的 `config.json`、`effective_config.json` 以及机器人 YAML/XML；
以对应 checkpoint 的有效配置为准，不能直接套用当前仓库默认值。

## 2. 维度约定和配置位置

| 符号 | 含义 |
|---|---|
| `L` | 输入参考状态序列的帧数 |
| `N` | Actor 推理批量大小，单机器人通常为 1 |
| `D` | 受控关节数量，G1 通常为 29，DR02 为 23 |
| `K` | 参与参考观测的 body 数量，包含根部及配置中的扩展节点 |
| `Z` | 潜变量维度，从 `archi.z_dim` 读取，例如 256 |
| `H` | 历史长度，由训练配置决定；不同字段可以使用不同长度 |

所有输入均需要显式批量/时间维度，使用 float32；不传一维 `(D,)` 向量。
脚本会将输入转换为 float32 并放到指定设备。

查看模型配置中的准确设置：

| 内容 | 文件及字段 |
|---|---|
| FB 编码器字段顺序 | `model/config.json → archi.b.input_filter.key` |
| TeCH 编码器字段顺序 | `model/config.json → archi.goal_encoder.input_filter.key` |
| Actor 字段顺序 | `model/config.json → archi.actor.input_filter.key` |
| z 维度 | `model/config.json → archi.z_dim` |
| 观测归一化 | `model/config.json → obs_normalizer.normalizers` |
| 各观测字段维度 | `model/init_kwargs.json → obs_space.spaces.<字段>.shape` |
| 动作维度 | `model/init_kwargs.json → action_dim` |

下面以仓库默认的字典输入配置为例；自定义 checkpoint 应以其 input_filter 为准。

## 3. B / encode 的输入

默认调用形式：

```python
ref_action = {
    'state': reference_state,                    # (L, 2*D+6)
    'privileged_state': reference_privileged,    # (L, P)
}
z = policy.encode(ref_action)                   # (L, Z)
```

默认 B 按 `state`、`privileged_state` 顺序拼接，不使用 `last_action` 或 `history_actor`。
输入描述的是**参考机器人状态**，而不是当前真实机器人的状态。

### 3.1 state：本体状态

严格按以下顺序拼接，再送入网络归一化器：

| 顺序 | 特征 | 维度 | 单位/坐标约定 |
|---|---|---|---|
| 1 | `q_ref - q_default` | D | rad，按训练的 control_joints 顺序 |
| 2 | `dq_ref` | D | rad/s，与关节角顺序一致 |
| 3 | `projected_gravity` | 3 | 无量纲，将世界向下单位重力向量转到完整根部局部坐标系 |
| 4 | `base_ang_vel` | 3 | rad/s，根部角速度转到完整根部局部坐标系 |

G1 的维度通常为 `29 + 29 + 3 + 3 = 64`。
每个子字段还需乘以训练配置中的 `obs.obs_scales.<字段名>`。
不要无条件把角速度除以 4；具体缩放取决于该 checkpoint 的训练配置。
参考观测不添加观测噪声。

### 3.2 privileged_state：完整身体状态

构造前需要以下量：

| 原始特征 | 形状 | 单位/说明 |
|---|---|---|
| `body_pos` | (L, K, 3) | 世界坐标，m，body/link 原点位置 |
| `body_rot` | (L, K, 4) | 世界姿态四元数，UFO 此处使用 XYZW |
| `body_vel` | (L, K, 3) | 世界坐标线速度，m/s，对应 body/link 原点 |
| `body_ang_vel` | (L, K, 3) | 世界坐标角速度，rad/s |

body 顺序必须与训练一致，索引 0 为根部。
令 `R_heading` 为根部朝向的旋转矩阵，只包含绕世界 z 轴的 yaw。
按照下面顺序构造并展平：

| 顺序 | 特征 | 维度 | 构造规则 |
|---|---|---|---|
| 1 | 根部高度，可选 | 1 或 0 | `root_height_obs=true` 时加入世界坐标根部 z |
| 2 | 局部 body 位置 | 3(K−1) | `R_heading⁻¹ (p_body − p_root)`，去掉根部位置项 |
| 3 | 局部 body 姿态 | 6K | `R_heading⁻¹ R_body` 转成 6 维表示，包含根部 |
| 4 | 局部 body 线速度 | 3K | `R_heading⁻¹ v_body`，包含根部；不减根部速度 |
| 5 | 局部 body 角速度 | 3K | `R_heading⁻¹ ω_body`，包含根部；不减根部角速度 |

6 维姿态表示是拼接旋转后的 **x 轴和 z 轴**：`[R e_x, R e_z]`，
即旋转矩阵的第一列和第三列，不是欧拉角、四元数或第一列和第二列。
每类特征内部按 body 顺序展平；不是逐 body 拼接所有特征。

因此：

```text
P = 15*K - 3 + (1 if root_height_obs else 0)
```

最后整体乘以 `obs.obs_scales.max_local_self`。
这里的 heading 坐标系与 `state` 中使用完整根部姿态的局部坐标系不同。
根部 roll/pitch 信息仍保留，不能全部消除。

### 3.3 encode 的输出行为

计算链路：

```text
参考状态特征（已完成几何处理和观测缩放）
→ checkpoint 的 BatchNorm/Identity 归一化
→ 按编码器 input_filter 拼接
→ B（或 TeCH goal encoder）
→ checkpoint 对应的 project_z
→ (L, Z)
```

FB 在 `norm_z=true` 时将 z 投影到半径 `sqrt(Z)` 的超球面（零向量除外）。
编码器内部是否也进行向量归一化由 `archi.b.norm` 决定。
TeCH 还支持配置中的 Poincaré ball 投影。

当前 `encode` 每行独立编码，不执行时间窗口聚合。原始 G1 输入会执行正运动学和速度计算；
可用 `sample_times` 显式重采样、`drop_first=True` 删除首帧，默认均不执行。
原仓库 tracking 入口会先按控制时间插值，再将 `backward_obs[1:]` 送入编码器，
所以要严格复现原流程，需要调用方保持对应的下一帧参考对齐。

## 4. 必须直接提供的输入与可以计算的量

本节以“从原始动作生成 B 输入，再做动作跟踪”为目标。
不要把第 3 节网络接收的所有特征都理解成需要独立采集的数据。

### 4.1 每条参考动作必须直接提供：四项

| 必须提供的原始数据 | G1 形状 | 说明 |
|---|---|---|
| 关节角 `dof_pos` | (L, 29) | rad，已重定向到 G1，顺序与机器人配置一致 |
| 根部位置 `root_pos` | (L, 3) | 世界坐标 XYZ，m，包含根部高度与移动轨迹 |
| 根部朝向 `root_quat` | (L, 4) | 有效四元数，顺序需明确并与配置一致；通常使用 XYZW |
| 采样率 `fps` | 正标量 | 或源读取器支持的时间列/时间数组，用于确定时间间隔 |

四项可以放在同一个文件中，不要求四个文件。关节角和根部位姿需要时间对齐。
如果文件中已经包含采样率，就不需要再通过命令行重复指定。
关节名称也不是每帧都必须传：可以在文件中保存一次 `joint_names`，
或明确约定全文件按模型配置的关节顺序排列。

对于标准 UFO PKL，这四项信息的等价存储方式是
`root_trans_offset + pose_aa + fps`：`pose_aa` 已包含根部与各身体节点的轴角姿态，
不必重复提供 `root_quat` 和 `dof_pos`。

**仅有 `(L,29)` 关节角无法唯一恢复根部位置和朝向。**
固定根部或手工假定高度只是近似，不等价于原始走路、跳跃、翻滚等动作。
绝对水平原点和整体 yaw 可以选择一致的约定，但不能丢掉轨迹随时间的平移和转向。

### 4.2 一次性提供的文件和配置

这些是元信息，不需要每条动作、每一帧重复输入。

| 必须准备的资源 | 从中读取的信息 |
|---|---|
| `model/config.json`、`init_kwargs.json`、`model.safetensors` | 网络架构、输入字段、维度、归一化统计和权重 |
| 匹配机器人的运动学模型，例如 XML | 连杆层级、关节轴、固定变换、body 布局，用于正运动学 |
| 与 checkpoint 匹配的机器人/有效训练配置 | 默认关节角、control_joints/body 排列、扩展节点、观测缩放、根部高度开关等 |
| 实际控制配置（用于 Actor 连续运行） | 控制周期、历史长度/顺序、动作缩放和截断规则 |

默认关节角、body 顺序等已有配置值应直接读取，不必让用户重新填写。
但它们不是从参考关节角序列本身推算出的量，也不能从网络维度反推出。
如果上游已经构造好了符合训练定义的网络特征，最小网络运行本身不需要再加载机器人模型。

### 4.3 不必额外直接提供：可以计算的量

| 可以计算的量 | 依据/处理 |
|---|---|
| 关节速度 `dof_vel` | 关节角序列 + FPS，按原流程进行时间差分 |
| 根部线速度 | 根部位置序列 + FPS |
| 根部角速度 | 根部姿态序列 + FPS，通过旋转差计算，不能直接差分四元数分量 |
| 每个 body 的位置与姿态 | 关节角、根部位姿 + 机器人模型，通过正运动学得到 |
| 每个 body 的线速度和角速度 | body 位姿序列 + FPS，按原流程差分/滤波 |
| 投影重力 | 根部姿态 + 世界向下单位重力向量 |
| 关节位置偏移 | 关节角 − 配置中的默认关节角 |
| heading 坐标变换及姿态 6D 表示 | 根部朝向 + body 位姿，规则见 3.2 |
| B 的 `state` | 以上关节/根部特征按 3.1 缩放、拼接 |
| B 的 `privileged_state` | 以上 body 特征按 3.2 变换、缩放、拼接 |
| 网络归一化后的输入 | checkpoint 保存的 BatchNorm 统计；最小脚本内部自动执行 |
| 任务潜变量 `z` | B/goal encoder + `project_z`；最小脚本的 `encode` 执行 |

已有匹配速度数据时可以保留原始速度；没有时 UFO 能计算。
如果要求数值复现，应匹配原仓库的差分、滤波、插值、坐标系和首帧对齐规则，
不能任意选择另一种速度估计方法。

### 4.4 Actor 当前状态：与参考动作分开准备

参考动作描述“希望机器人做什么”，不能代替“机器人当前是什么状态”。
真实机器人每个控制步需要当前关节角、关节速度、根部姿态/重力方向、根部角速度，
通常来自关节反馈和 IMU/状态估计；原 UFO 仿真流程由仿真器提供这些当前状态。
关节速度和角速度也可以由带时间信息的当前状态历史估计，但需处理噪声和延迟，
**不能拿参考轨迹计算出的速度代替当前机器人的速度**。

以下 Actor 量不需要另外采集：

- `state`：由当前关节与 IMU/状态估计数据按第 5 节构造。
- `last_action`：程序记录上一步动作，按训练时的动作处理与观测缩放构造。
- `history_actor`：程序维护历史缓存，按训练时的长度和顺序排列。
- `z`：使用参考动作编码结果。

默认 Actor 不使用完整 body 特征，所以不需要实时测量所有 body 的位置和速度；
是否有额外字段仍以该 checkpoint 的 Actor input_filter 为准。

### 4.5 原 UFO 与当前最小脚本的分工

| 环节 | 原 UFO 完整 tracking 流程 | 当前 `inference_minimal/infer.py` |
|---|---|---|
| 原始 CSV/NPZ 转换、PKL 读取 | 有数据工具和 MotionLib | 未实现 |
| 正运动学、参考速度及特征构造 | 自动计算 | 由独立的 `UFO_observation_compute` 模块提供 |
| 当前机器人状态与历史 | 仿真环境维护 | 需要调用方提供/维护 |
| 保存统计量的观测归一化 | 自动执行 | 自动执行 |
| B 编码及 z 投影 | 自动执行 | `policy.encode(ref_action)` |
| Actor 推理 | 自动执行 | `policy(obs, z)` |

第 4.1 节是准备**原始参考动作**的最小清单，G1 可直接传入 `encode`。
也可以继续使用第 3 节的 `state`、`privileged_state` 特征字典。
正运动学预处理已整理至 [UFO_observation_compute](UFO_observation_compute/README.md)，
仅依赖 pip 可安装的 Torch、NumPy、SciPy，无需仿真器。
`encode` 可通过 `UFO_observation.py` 自动调用该模块，也支持手动生成特征字典再编码。

### 4.6 原仓库可能遇到的动作文件

- 标准 UFO `*.pkl`：以动作名为键的字典，必需字段为
  `root_trans_offset: (L,3)`、`pose_aa: (L,K,3)`、`fps`。
  `pose_aa` 是按身体节点排列的轴角姿态，包含根部姿态，不能当作 `(L,D)` 关节角。
  还可能包含 `dof_pos`、`dof_vel`、`body_vel`、`body_ang_vel`、关节/body 名称等。
- RobotState `*.npz`：通常包含 `root_pos`、`root_quat`、`dof_pos`、`fps`/`time`，
  推荐提供 `joint_names`。四元数顺序应与机器人数据配置一致。
- RobotState `*.csv`：无表头形式通常为根部 XYZ、四元数 XYZW、D 个关节角，
  可带前导时间列。无时间信息时需显式提供 FPS；关节顺序不可猜测。
- `configs/data/*.yaml`：数据清单，描述来源、格式、FPS、切片等；不是网络输入。
- `*_full_ufo.pkl`：转换后的完整动作，通常用于跟踪推理。
- `*_train_ufo.pkl` 或 `*_10s-clipped.pkl`：训练片段，动作 ID 与完整动作不同。
- `tracking_inference/zs_<motion_id>.pkl`：原仓库已经编码得到的 z 序列，不是原始关节角。

最小 `infer.py` 不直接读取这些动作文件；它接收已经构造好的参考观测张量。

## 5. Actor 的输入

默认 Actor 字段为 `state`、`last_action`、`history_actor`，以及单独传入的 z：

| 输入 | 形状 | 来源 |
|---|---|---|
| `obs['state']` | (N, 2D+6) | 当前机器人状态，特征顺序同 3.1 |
| `obs['last_action']` | (N, D) | 上一控制步经原环境处理并应用观测缩放后的动作 |
| `obs['history_actor']` | (N, Dhistory) | 历史本体观测与动作，按训练配置排列 |
| `z` | (N, Z) | `encode` 得到的目标潜变量，或外部提供的兼容潜变量 |

这里的 `state` 是**当前机器人状态**；编码器的 `state` 是**参考状态**。
Actor 的关节位置偏移还应匹配有效默认姿态；若环境使用了默认姿态随机偏移，
需要与其处理一致，不能混用未偏移的参考默认姿态。
默认 Actor 不用 privileged_state；其他配置以其 input_filter 为准。

### last_action 的注意事项

UFO 环境对 Actor 的 tanh 输出可能先进行 `normalize_action_to / normalize_action_from`
缩放，然后按 `action_clip_value` 截断，并乘观测的 `actions` scale。
因此 **last_action 不一定等于未经处理的上次网络输出**，也不是实际关节角或电机力矩。
动作缩放参数须从对应实验配置读取。

### history_actor 的注意事项

原代码按历史字段名称排序后拼接，默认顺序是：

```text
actions → base_ang_vel → dof_pos → dof_vel → projected_gravity
```

对每个字段先取其 H 帧历史并展平，再拼接下一个字段；不是逐帧拼整个 state。
`history_oldest_first=false` 时，同一字段内从新到旧排列；为 true 时从旧到新排列。
历史来自已进行子字段观测缩放的值，拼接后还乘 `history_actor` scale。
历史先读取，再插入当前帧，所以不应把当前帧重复放进历史。

若所有字段均使用 H=4，G1 的历史维度为：

```text
Dhistory = 4 * (29 + 3 + 29 + 29 + 3) = 372
```

默认 Actor 拼接观测宽度此时为 `64 + 29 + 372 = 465`，z 是额外输入。
这些是示例维度，必须与 checkpoint 的 shape 一致。

## 6. 两层预处理不能混淆

参考动作的默认姿态偏移、坐标变换、速度计算和 observation scales
可由 `UFO_observation_compute` 完成；调用方需要提供正确的机器人模型、
关节顺序和训练配置。当前机器人状态的历史排列、动作历史处理仍由调用方负责。

`Policy` 内部负责：使用 checkpoint 的 BatchNorm 运行均值/方差（或 Identity）归一化，
然后执行网络。**不要在外部重复做这层 BatchNorm。**
纯网络零输入运行不要求机器人资源；从真实动作和传感器生成正确特征则需要上述元信息。

## 7. 安装和运行

```bash
python -m pip install -r inference_minimal/requirements.txt

# 全零观测 + 全零 z → Actor
python inference_minimal/infer.py \
  --checkpoint /path/to/checkpoint --device cpu

# 全零参考观测 → B/goal encoder → z；全零当前观测 + z → Actor
python inference_minimal/infer.py \
  --checkpoint /path/to/checkpoint --device cpu --encode-reference
```

`--checkpoint` 接受实验根目录、checkpoint 目录或 model 目录。
GPU 使用 `--device cuda:0`，批量输入使用 `--batch-size 4`。
零参考输入编码后的 z、零观测推理后的动作均不保证为零。
这些输入仅用于检查网络运行，不是有效的参考动作或真机控制输入。

## 8. Python 串联示例

在 `inference_minimal/` 目录中运行，下面使用默认字典观测 checkpoint：

```python
import torch
from infer import Policy

policy = Policy('/path/to/checkpoint', device='cpu')
L = 100

# 这里只是零输入示例。真实跟踪时替换为第 3 节定义的参考特征。
# 自动创建所有观测字段，encode 只取配置要求的字段。
ref_action = {
    key: torch.zeros(L, dim, device=policy.device)
    for key, dim in policy.obs_dims.items()
}
z_sequence = policy.encode(ref_action)   # (L, Z)

# 示例批量大小为 1；实际使用时每步填写当前传感器状态、动作历史和观测历史。
obs, _ = policy.zero_inputs(batch_size=1)
z_t = z_sequence[0:1]                    # 保留 batch 维度 (1, Z)
action = policy(obs, z_t)                # (1, D)
print('z:', z_t.shape, 'action:', action.shape)
```

Actor 输出与 UFO `model.act(obs, z, mean=True)` 对应，是 [-1,1] 范围的归一化动作。
这里不执行 PD、不发送控制命令，也不将输出转换成关节目标角或力矩。
完整控制还需要与训练一致的动作到关节目标映射和电机参数。

## 9. 原仓库对应代码

- `ufo/motion/reference.py`：参考 state/privileged_state 的生成。
- `ufo/envs/observations.py`：坐标变换、姿态六维表示、特征和历史拼接。
- `ufo/envs/core.py`：当前机器人观测及动作处理。
- `ufo/motion/library.py`、`kinematics.py`：动作读取、正运动学及速度计算。
- `ufo/agents/common/inputs.py`：观测过滤及 BatchNorm。
- `ufo/agents/common/networks.py`：BackwardMap 和 Actor 网络。
- `ufo/agents/fb_cpr_aux/model.py`、`ufo/agents/tech/model.py`：潜变量投影及编码接口。
- `ufo/inference/tracking_inference.py`：参考帧对齐、z 保存及跟踪调用流程。

网络结构改写自 UFO，保留上游 CC BY-NC 4.0 许可，见 LICENSE。
