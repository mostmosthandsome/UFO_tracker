"""DR02 asset, motion-order and environment integration regression checks."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
import yaml
from easydict import EasyDict
from scipy.spatial.transform import Rotation

from ufo.motion.formats import RobotStateMotion, robot_state_to_ufo_motion
from ufo.motion.kinematics import HumanoidKinematics
from ufo.tools.robot_config import load_robot_training_spec

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/robots/dr02_23dof.yaml"


def test_dr02_layout_and_source_control_parameters():
    spec = load_robot_training_spec(CONFIG)
    model = mujoco.MjModel.from_xml_path(str(spec.robot.xml_file))
    assert (model.nq, model.nv, model.nu) == (30, 29, 23)
    assert "actuatorfrcrange" not in spec.robot.xml_file.read_text()
    assert spec.action_scale == 0.25 and not spec.action_rescale
    assert spec.default_joint_angles["left_knee_joint"] == 0.2
    assert spec.stiffness["left_hip_y_joint"] == 250
    assert spec.damping["right_ankle_x_joint"] == 1
    assert spec.actuator["joints"]["waist_y_joint"]["effort_limit"] == 413
    assert spec.actuator["joints"]["left_elbow_joint"]["armature"] == 0.005
    assert spec.robot.control_joint_names == spec.robot.actuator_joint_names
    assert all("wrist" not in name for name in spec.robot.control_joint_names)


def test_dr02_mjcf_preserves_urdf_forward_kinematics():
    """Independent URDF traversal catches lost fixed links, axes and origins."""
    spec = load_robot_training_spec(CONFIG)
    model = mujoco.MjModel.from_xml_path(str(spec.robot.xml_file))
    data = mujoco.MjData(model)
    urdf = ET.parse(spec.robot.xml_file.parent / "dr02_pro.urdf")
    rng = np.random.default_rng(42)
    for _ in range(3):
        angles = dict(zip(spec.robot.control_joint_names, rng.uniform(-0.3, 0.3, 23)))
        data.qpos[:7] = [0, 0, 0.93, 1, 0, 0, 0]
        for name, angle in angles.items():
            data.qpos[spec.robot.joint_qpos_addr[name]] = angle
        mujoco.mj_forward(model, data)
        transforms = {"base_link": np.eye(4)}
        transforms["base_link"][2, 3] = 0.93
        pending = list(urdf.findall("joint"))
        while pending:
            ready = [j for j in pending if j.find("parent").get("link") in transforms]
            assert ready, "Invalid URDF tree"
            for joint in ready:
                origin = joint.find("origin")
                local = np.eye(4)
                if origin is not None:
                    local[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                    local[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
                if joint.get("type") != "fixed":
                    axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                    local[:3, :3] = local[:3, :3] @ Rotation.from_rotvec(axis * angles[joint.get("name")]).as_matrix()
                transforms[joint.find("child").get("link")] = transforms[joint.find("parent").get("link")] @ local
                pending.remove(joint)
        for name, transform in transforms.items():
            body_id = model.body(name).id
            np.testing.assert_allclose(data.xpos[body_id], transform[:3, 3], atol=1e-5)
            np.testing.assert_allclose(data.xmat[body_id].reshape(3, 3), transform[:3, :3], atol=1e-5)


def test_dr02_named_motion_and_ufo_fk_match_mujoco():
    spec = load_robot_training_spec(CONFIG).robot
    names = list(reversed(spec.control_joint_names))
    positions = np.random.default_rng(7).uniform(-0.2, 0.2, (5, 23)).astype(np.float32)
    state = RobotStateMotion("test", np.tile([0, 0, 0.93], (5, 1)),
                             np.tile([0, 0, 0, 1], (5, 1)), positions, 30, joint_names=names)
    motion = robot_state_to_ufo_motion(state, spec)
    np.testing.assert_array_equal(motion["dof_pos"], positions[:, ::-1])
    kin = HumanoidKinematics(EasyDict(asset=EasyDict(assetRoot=str(spec.xml_file.parent),
                                                    assetFileName=spec.xml_file.name), extend_config=[]))
    fk = kin.fk_batch(torch.from_numpy(motion["pose_aa"][None]),
                      torch.from_numpy(motion["root_trans_offset"][None]), dt=1 / 30)
    model = mujoco.MjModel.from_xml_path(str(spec.xml_file))
    data = mujoco.MjData(model)
    for frame in range(5):
        data.qpos[:7] = [0, 0, 0.93, 1, 0, 0, 0]
        for idx, name in enumerate(spec.control_joint_names):
            data.qpos[spec.joint_qpos_addr[name]] = motion["dof_pos"][frame, idx]
        mujoco.mj_forward(model, data)
        expected = np.array([data.xpos[model.body(name).id] for name in kin.body_names])
        np.testing.assert_allclose(fk.global_translation[0, frame].numpy(), expected, atol=2e-6)


def test_dr02_native_task_action_scale():
    from ufo.envs.config import MJLabEnvConfig, _action_target_scale_values, resolve_task_config

    spec = load_robot_training_spec(CONFIG)
    env = MJLabEnvConfig(motion_data_path="unused.pkl", robot_training=spec.to_env_dict(), task_overrides=spec.task_overrides)
    task = resolve_task_config(env, 2)
    assert task.robot.actions_dim == 23
    assert _action_target_scale_values(task) == [0.25] * 23
    assert task.robot.torso_name == "body"


def test_dr02_csv_import_uses_source_joint_order(tmp_path, monkeypatch):
    from ufo.tools.dr02_import import import_motions

    config = yaml.safe_load(CONFIG.read_text())
    config["xml_path"] = str(ROOT / config["xml_path"])
    config_dir = tmp_path / "configs/robots"
    config_dir.mkdir(parents=True)
    (config_dir / CONFIG.name).write_text(yaml.safe_dump(config))
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    names = list(reversed(config["control_joints"]))
    (source / "scripts/csv_to_npz.py").write_text(f"run_simulator(sim, scene, joint_names={names!r})")
    values = np.zeros((60, 30))
    values[:, 2] = 0.93
    values[:, 6] = 1
    values[:, 7:] = np.arange(23) * 0.01
    csv = source / "test.csv"
    np.savetxt(csv, values, delimiter=",")
    monkeypatch.chdir(tmp_path)
    import_motions(source, [csv], 30)
    with np.load("cache/dr02_robot_state/test.npz") as motion:
        assert motion["joint_names"].tolist() == config["control_joints"]
        np.testing.assert_allclose(motion["dof_pos"], values[:, 7:][:, ::-1])
        assert float(motion["fps"]) == 30
    assert Path("configs/data/dr02_local.yaml").exists()
