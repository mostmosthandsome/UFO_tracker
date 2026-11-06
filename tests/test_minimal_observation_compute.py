"""Compare standalone reference preprocessing with the original UFO functions."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from easydict import EasyDict
from scipy.spatial.transform import Rotation

from inference_minimal.UFO_observation_compute import ReferenceObservationComputer
from ufo.motion.kinematics import HumanoidKinematics
from ufo.motion.reference import _motion_observations

ROOT = Path(__file__).resolve().parents[1]


class ObservationComputeTest(unittest.TestCase):
    def test_reference_features_match_ufo(self):
        for robot in ("g1_29dof", "dr02_23dof"):
            for root_height in (False, True):
                with self.subTest(robot=robot, root_height=root_height):
                    config = yaml.safe_load((ROOT / f"configs/robots/{robot}.yaml").read_text())
                    names = config["control_joints"]
                    if isinstance(names, dict):
                        self.assertEqual(names["mode"], "all_actuated")
                        names = [node.attrib["joint"] for node in
                                 ET.parse(ROOT / config["xml_path"]).findall("./actuator/*")]
                    training = config["training"]
                    defaults = [training["init_state"]["default_joint_angles"][name] for name in names]
                    scales = {"dof_pos": 0.8, "dof_vel": 0.2, "base_ang_vel": 0.25,
                              "projected_gravity": 0.9, "max_local_self": 0.7}
                    metadata = dict(control_joint_names=names, default_dof_pos=defaults,
                                    motion_extensions=training.get("motion_extensions", []),
                                    root_height_obs=root_height, obs_scales=scales)
                    xml = ROOT / config["xml_path"]
                    computer = ReferenceObservationComputer(xml, **metadata)
                    rng = np.random.default_rng(12)
                    q = torch.tensor(rng.normal(0, 0.1, (12, len(names))), dtype=torch.float32)
                    trans = torch.tensor(rng.normal(0, 0.1, (12, 3)), dtype=torch.float32)
                    trans[:, 2] += 0.9
                    rotvec = rng.normal(0, 0.1, (12, 3))
                    quat = Rotation.from_rotvec(rotvec).as_quat().astype(np.float32)
                    fps = 30
                    original = HumanoidKinematics(EasyDict(
                        asset=EasyDict(assetRoot=str(xml.parent), assetFileName=xml.name),
                        extend_config=[EasyDict(item) for item in metadata["motion_extensions"]],
                    ))
                    pose = torch.zeros(12, original.num_bodies, 3)
                    pose[:, 0] = torch.from_numpy(Rotation.from_quat(quat).as_rotvec()).float()
                    for i, name in enumerate(names):
                        body, axis = computer.joints[name]
                        pose[:, original.body_names.index(body)] = q[:, i:i+1] * torch.tensor(axis, dtype=torch.float32)
                    fk = original.fk_batch(pose[None], trans[None], dt=1/fps)
                    suffix = "_extend" if metadata["motion_extensions"] else ""
                    dq = torch.diff(q, dim=0) * fps
                    motion = {"rg_pos_t": getattr(fk, "global_translation"+suffix)[0],
                              "rg_rot_t": getattr(fk, "global_rotation"+suffix)[0],
                              "body_vel_t": getattr(fk, "global_velocity"+suffix)[0],
                              "body_ang_vel_t": getattr(fk, "global_angular_velocity"+suffix)[0],
                              "dof_vel": torch.cat((dq, dq[-1:]))}
                    env = SimpleNamespace(config=SimpleNamespace(obs=SimpleNamespace(obs_scales=scales)),
                                          gravity_vec=torch.tensor([[0., 0., -1.]]))
                    state, privileged = _motion_observations(env, motion, q-torch.tensor(defaults),
                                                             root_height_obs=root_height, use_obs_filter=True)
                    # Deliberately reverse source joint order, exercise name-based mapping.
                    result = computer.from_robot_state(q.flip(1), trans, quat, fps, joint_names=names[::-1])
                    torch.testing.assert_close(result["state"], state, atol=2e-5, rtol=2e-5)
                    torch.testing.assert_close(result["privileged_state"], privileged, atol=2e-5, rtol=2e-5)
                    self.assertEqual(result["privileged_state"].shape[1], 15*len(computer.body_names)-3+int(root_height))
                    # Interpolation and next-frame shift must preserve source-aligned frames.
                    sampled = computer.from_robot_state(q, trans, quat, fps,
                                                        sample_times=torch.tensor([0., 1/fps, 2/fps]), drop_first=True)
                    torch.testing.assert_close(sampled["state"], result["state"][1:3], atol=2e-5, rtol=2e-5)
                    torch.testing.assert_close(sampled["privileged_state"], result["privileged_state"][1:3], atol=2e-5, rtol=2e-5)
                    for length in (1, 2):
                        short = computer.from_robot_state(q[:length], trans[:length], quat[:length], fps)
                        self.assertEqual(short["state"].shape[0], length)
                        self.assertTrue(all(torch.isfinite(value).all() for value in short.values()))
                    with self.assertRaises(ValueError):
                        computer.from_robot_state(q, trans, np.zeros_like(quat), fps)
                    if robot == "dr02_23dof" and root_height:
                        self._check_standalone_cli(xml, metadata, q, trans, quat)

    def _check_standalone_cli(self, xml, metadata, q, trans, quat):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            shutil.copytree(ROOT / "inference_minimal/UFO_observation_compute", path / "UFO_observation_compute",
                            ignore=shutil.ignore_patterns("__pycache__"))
            (path / "robot.json").write_text(json.dumps(metadata))
            np.savez(path / "motion.npz", dof_pos=q.numpy(), root_pos=trans.numpy(), root_quat=quat, fps=30)
            environment = dict(os.environ)
            environment.pop("PYTHONPATH", None)
            result = subprocess.run([sys.executable, "-m", "UFO_observation_compute", "--xml", str(xml),
                                     "--metadata", "robot.json", "--motion", "motion.npz", "--output", "obs.pt"],
                                    cwd=path, env=environment, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((path / "obs.pt").is_file())


if __name__ == "__main__":
    unittest.main()
