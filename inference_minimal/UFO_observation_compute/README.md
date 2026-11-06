# 从参考动作计算 UFO 观测

本目录可以独立复制使用，只依赖 pip 安装的 `torch`、`numpy`、`scipy`。
无需导入 UFO、mjlab、MuJoCo 或 IsaacLab，也不执行物理仿真。
计算对象是**参考动作**，输出可传给 `infer.py` 的 `Policy.encode()`。

## 文件

| 文件 | 作用及原始来源 |
|---|---|
| `reference.py` | 输入检查、关节映射、参考采样和观测构造；对应 `ufo/motion/reference.py`、`formats.py`、`library.py` 的相关流程 |
| `kinematics.py` | XML 骨架解析、正运动学和速度滤波；提取自 `ufo/motion/kinematics.py` |
| `observations.py` | 特权信息拼接；提取 `ufo/envs/observations.py` 中的 `compute_humanoid_observations_max` |
| `math_utils.py` | 所需四元数、旋转及插值函数；提取自 `ufo/math.py`，保留原版权声明 |
| `__init__.py` | Python 接口 |
| `__main__.py` | NPZ 转观测字典的命令行入口 |
| `requirements.txt` | 本模块依赖 |

已附带 `g1_29dof.xml` 和 `g1_metadata.json`，对应仓库 G1 默认设置，供上一级
`UFO_observation.py` 和 `Policy.encode` 使用。其他配置需要另外提供，必须与 checkpoint 对应。XML 只读取骨架，
不读取 mesh，因此不需要复制 meshes、碰撞模型资源或仿真器配置。

## 必须提供的原始数据

| 输入 | 形状/含义 |
|---|---|
| `dof_pos` | `(L,D)`，关节角，单位 rad；G1 为 `(L,29)` |
| `root_pos` | `(L,3)`，根节点世界坐标，单位 m，Z 向上 |
| `root_quat` | `(L,4)`，根节点世界姿态；默认 XYZW |
| `fps` | 原始动作采样频率，正数 |
| `joint_names` | 可选，输入关节列名；不提供时必须已按 `control_joint_names` 排列 |

只有关节角不能确定根节点的运动，不能据此恢复完整特权信息。
Python 接口支持 `quat_order="wxyz"`；命令行 NPZ 输入须使用 XYZW。

必须配置的机器人/观测元信息：

- `xml_path`：训练机器人使用的显式 MJCF 骨架。
- `control_joint_names`：checkpoint 对应的关节顺序，不能随意改用字母顺序。
- `default_dof_pos`：按该顺序排列的训练默认关节角。
- `root_height_obs`：训练时是否包含根部高度。
- `obs_scales`：训练观测缩放系数，包括 `dof_pos`、`dof_vel`、
  `projected_gravity`、`base_ang_vel`、`max_local_self`；缺省键按原流程取 1。
- `motion_extensions`：训练参考骨架的扩展节点；无扩展时为 `[]`。
  G1 的配置包含虚拟 `head_link`，必须保留，否则特权观测维度/语义不匹配。

上述参数应来自实际训练配置；`checkpoint/model/config.json` 的网络维度不足以还原它们。
仓库的 `configs/robots/*.yaml` 可用于查机器人配置，但运行时覆盖值以实际训练配置为准。
扩展节点格式为 `{"joint_name": ..., "parent_name": ..., "pos": [x,y,z], "rot": [w,x,y,z]}`。

## 自动计算的量

1. 各 body 的世界位置、世界姿态（正运动学，包含扩展节点）。
2. 关节角速度，以及各 body 的线速度、角速度；保留原 UFO 的 body 速度高斯滤波规则。
3. 根部局部角速度、投影重力、默认姿态偏移和观测缩放。
4. heading 坐标系下的 body 相对位置、6D 旋转、线速度、角速度，以及可选根部高度。

输出字典：

| 键 | 形状/顺序 |
|---|---|
| `state` | `(N,2*D+6)`：关节角偏移、关节速度、投影重力、局部根角速度 |
| `privileged_state` | `(N,15*K-3+H)`：可选根高度、相对 body 位置（去掉根）、所有 body 的 6D 旋转、线速度、角速度 |
| `last_action` | `(N,D)`，参考观测中的零占位；不是 Actor 的真实历史动作 |

其中 `K` 包括扩展节点，`H=int(root_height_obs)`。默认 `N=L`。
body 顺序可从 `computer.body_names` 查看，遵循 XML 树及扩展节点顺序。
6D 旋转使用旋转矩阵的第一、第三列；速度仅旋转到 heading 坐标系，不减去根速度。
这里只做训练配置中的尺度缩放，不做 checkpoint 内的 BatchNorm；后者由 `Policy` 完成。

若动作已提供精确导数，可传可选 `dof_vel: (L,D)`、`body_vel: (L,K,3)`、
`body_ang_vel: (L,K,3)` 覆盖差分结果。body 导数必须按 `computer.body_names` 排列，
使用世界坐标系、节点原点速度（不是质心速度），包含扩展节点。

## Python 使用

在 `inference_minimal/` 目录运行：

```bash
python -m pip install -r UFO_observation_compute/requirements.txt
```

准备 `robot_metadata.json`，内容是上述构造参数（不含 `xml_path`）：
`control_joint_names`、`default_dof_pos`、`root_height_obs`、`obs_scales`、`motion_extensions`。
列表长度、顺序和系数均须填写实际训练值。

```python
import json
import numpy as np
from UFO_observation_compute import ReferenceObservationComputer
from infer import Policy

metadata = json.load(open("robot_metadata.json"))
computer = ReferenceObservationComputer("/path/to/robot.xml", **metadata)
with np.load("motion.npz", allow_pickle=False) as motion:
    ref_obs = computer.from_robot_state(
        dof_pos=motion["dof_pos"],
        root_pos=motion["root_pos"],
        root_quat=motion["root_quat"],
        fps=float(motion["fps"]),
        joint_names=motion["joint_names"].tolist() if "joint_names" in motion else None,
    )
policy = Policy("/path/to/checkpoint", device="cpu")
z = policy.encode(ref_obs)
```

网络推理另需安装上一级 `requirements.txt` 中的 `safetensors`。
`encode` 支持特征字典，也支持 G1 关节角加 `root_pos`、`root_quat`、`fps`；
后一种方式通过上一级 `UFO_observation.py` 在内部调用本模块。
已有全部 body 运动学结果时，也可直接使用导出的 `compute_reference_observation()`，跳过 XML/FK。

## 命令行

```bash
python -m UFO_observation_compute \
  --xml /path/to/robot.xml \
  --metadata robot_metadata.json \
  --motion motion.npz \
  --output ref_observations.pt
```

NPZ 至少包含 `dof_pos`、`root_pos`、`root_quat`、`fps`。
可以用 `--fps` 覆盖采样频率；关节名称用 Unicode 字符串数组保存，不使用 object 数组。
输出可通过 `torch.load("ref_observations.pt", weights_only=True)` 加载，再传给 `policy.encode`。

原 UFO tracking 先在控制时间轴上插值，再取下一帧参考 `[1:]`。
需要复现该对齐时增加 `--control-fps <实际控制频率> --drop-first`。
Python 对应 `sample_times=<秒为单位的一维时间数组>, drop_first=True`。
命令行控制时间点为 `arange(ceil(duration*control_fps))/control_fps`，不包含源动作末端。
默认不重采样、不丢首帧，便于直接编码整段参考动作。

## 支持范围与验证

这是原 UFO XML FK 所支持的骨架计算，不是完整 MJCF 编译器：需要单一根节点、
显式 body `pos/quat`、每个 body 至多一个位于原点的 hinge；不展开 XML include，
也不解析隐式几何默认值。其他机器人应先转换成这种显式骨架。
目前命令行读取 RobotState NPZ，不包含原仓库的 PKL/CSV 数据加载器。

`tests/test_minimal_observation_compute.py` 将 G1、DR02 的输出与原 UFO FK 和参考观测函数进行数值对照，
覆盖根高度开关、关节重排、短序列、采样对齐及脱离仓库的命令行运行。
