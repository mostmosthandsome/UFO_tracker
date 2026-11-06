"""Raw reference joint/root trajectories -> UFO B observation features.

Adapted from ufo/motion/reference.py, formats.py and library.py. CPU FK/velocity
preprocessing follows UFO; final observation tensors may be returned on CUDA.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .kinematics import HumanoidKinematics
from .math_utils import quat_rotate_inverse, slerp
from .observations import compute_humanoid_observations_max


def _tensor(value, name, shape):
    value = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if tuple(value.shape) != tuple(shape) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape {tuple(shape)}, got {tuple(value.shape)}")
    return value


@torch.inference_mode()
def compute_reference_observation(
    dof_pos, dof_vel, body_pos, body_rot, body_vel, body_ang_vel, default_dof_pos,
    *, root_height_obs: bool, obs_scales: dict, device: str = "cpu",
):
    """Already computed reference kinematics -> state/privileged_state/zero actions.

    Bodies are ordered root first; world rotations are XYZW and velocities are
    world-frame link-origin velocities, not COM velocities. No network BatchNorm.
    """
    dof_pos = torch.as_tensor(dof_pos, dtype=torch.float32, device=device)
    if dof_pos.ndim != 2 or min(dof_pos.shape) < 1:
        raise ValueError("dof_pos must have shape [L, D] with L,D >= 1")
    length, joints = dof_pos.shape
    body_pos = torch.as_tensor(body_pos, dtype=torch.float32, device=device)
    if body_pos.ndim != 3 or body_pos.shape[0] != length or body_pos.shape[1] < 1 or body_pos.shape[2] != 3:
        raise ValueError("body_pos must have shape [L, K, 3]")
    bodies = body_pos.shape[1]
    def checked(value, name, shape):
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        if tuple(tensor.shape) != tuple(shape) or not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be finite with shape {shape}")
        return tensor
    dof_pos = checked(dof_pos, "dof_pos", (length, joints))
    dof_vel = checked(dof_vel, "dof_vel", (length, joints))
    default = checked(default_dof_pos, "default_dof_pos", (joints,))
    body_pos = checked(body_pos, "body_pos", (length, bodies, 3))
    body_rot = checked(body_rot, "body_rot", (length, bodies, 4))
    norms = body_rot.norm(dim=-1, keepdim=True)
    if (norms < 1e-8).any():
        raise ValueError("body_rot contains zero quaternions")
    body_rot = body_rot / norms
    body_vel = checked(body_vel, "body_vel", (length, bodies, 3))
    body_ang_vel = checked(body_ang_vel, "body_ang_vel", (length, bodies, 3))
    def scale(name, value):
        factor = float(obs_scales.get(name, 1.0))
        if not math.isfinite(factor):
            raise ValueError(f"Invalid observation scale: {name}")
        return value * factor
    privileged = compute_humanoid_observations_max(
        body_pos, body_rot, body_vel, body_ang_vel, True, root_height_obs,
    )
    root_quat = body_rot[:, 0]
    gravity = torch.tensor([0., 0., -1.], device=device).expand(length, -1).contiguous()
    state = torch.cat([
        scale("dof_pos", dof_pos - default),
        scale("dof_vel", dof_vel),
        scale("projected_gravity", quat_rotate_inverse(root_quat, gravity, w_last=True)),
        scale("base_ang_vel", quat_rotate_inverse(root_quat, body_ang_vel[:, 0], w_last=True)),
    ], dim=-1)
    return {"state": state,
            "privileged_state": scale("max_local_self", torch.cat(list(privileged.values()), dim=-1)),
            "last_action": torch.zeros_like(dof_pos)}


class ReferenceObservationComputer:
    """Use an explicit UFO-style MJCF tree, without loading meshes or MuJoCo."""

    def __init__(self, xml_path, control_joint_names, default_dof_pos, *,
                 root_height_obs: bool, obs_scales: dict, motion_extensions=(), device="cpu"):
        self.control_joint_names = list(control_joint_names)
        if not self.control_joint_names or len(set(self.control_joint_names)) != len(self.control_joint_names):
            raise ValueError("control_joint_names must be nonempty and unique")
        self.default_dof_pos = _tensor(default_dof_pos, "default_dof_pos", (len(self.control_joint_names),))
        self.root_height_obs = root_height_obs
        self.obs_scales = dict(obs_scales)
        self.device = device
        path = Path(xml_path).expanduser().resolve()
        tree = ET.parse(path)
        if tree.findall(".//include"):
            raise ValueError("Use a self-contained UFO MJCF; XML includes are not expanded")
        roots = tree.findall("./worldbody/body")
        if len(roots) != 1:
            raise ValueError("Expected a single robot root body")
        # The upstream XML FK reads explicit pos/quat, not a MuJoCo compiler.
        geometry_defaults = [node for node in tree.findall("./default//joint")
                             if any(key in node.attrib for key in ("axis", "pos", "type"))]
        if tree.findall("./default//body") or geometry_defaults:
            raise ValueError("Resolve body/joint defaults into explicit MJCF attributes first")
        self.joints = {}
        for body in roots[0].iter("body"):
            if any(key in body.attrib for key in ("euler", "axisangle", "xyaxes", "zaxis")):
                raise ValueError("Body rotations must use explicit WXYZ quat attributes")
            joints = [j for j in body.findall("joint") if j.get("type", "hinge") != "free"]
            if len(joints) > 1:
                raise ValueError("Only one hinge joint per body is supported")
            for joint in joints:
                if joint.get("type", "hinge") != "hinge" or np.any(np.fromstring(joint.get("pos", "0 0 0"), sep=" ") != 0):
                    raise ValueError("Only hinge joints at body origins are supported")
                axis = np.fromstring(joint.get("axis", "0 0 1"), sep=" ")
                if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8:
                    raise ValueError(f"Invalid joint axis: {joint.get('name')}")
                self.joints[joint.get("name")] = (body.get("name"), axis / np.linalg.norm(axis))
        if set(self.joints) != set(self.control_joint_names):
            raise ValueError("control_joint_names must match the MJCF hinge joints exactly")
        extensions = [SimpleNamespace(**dict(item)) for item in motion_extensions]
        self.kinematics = HumanoidKinematics(SimpleNamespace(
            asset=SimpleNamespace(assetRoot=str(path.parent), assetFileName=path.name), extend_config=extensions,
        ))
        self.body_names = list(self.kinematics.body_names_augment)

    @torch.inference_mode()
    def from_robot_state(self, dof_pos, root_pos, root_quat, fps, *, joint_names=None,
                         quat_order="xyzw", sample_times=None, dof_vel=None,
                         body_vel=None, body_ang_vel=None, drop_first=False):
        """Encode-ready observations from [L,D], [L,3], [L,4], source FPS.

        sample_times optionally interpolates computed kinematics to control times
        (seconds relative to the first frame). drop_first reproduces tracking's
        next-frame alignment after interpolation. No physics step is performed.
        """
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        q = torch.as_tensor(dof_pos, dtype=torch.float32).detach().cpu()
        if q.ndim != 2 or q.shape[0] < 1:
            raise ValueError("dof_pos must have shape [L,D], L >= 1")
        length, count = q.shape
        q = _tensor(q, "dof_pos", (length, len(self.control_joint_names)))
        if joint_names is not None:
            names = list(joint_names)
            if len(names) != count or len(set(names)) != count or set(names) != set(self.control_joint_names):
                raise ValueError("joint_names must match control_joint_names exactly")
            order = [names.index(name) for name in self.control_joint_names]
            q = q[:, order]
            if dof_vel is not None:
                dof_vel = _tensor(dof_vel, "dof_vel", (length, count))[:, order]
        trans = _tensor(root_pos, "root_pos", (length, 3))
        quat = _tensor(root_quat, "root_quat", (length, 4))
        if quat_order == "wxyz":
            quat = quat[:, [1, 2, 3, 0]]
        elif quat_order != "xyzw":
            raise ValueError("quat_order must be xyzw or wxyz")
        if (quat.norm(dim=-1) < 1e-8).any():
            raise ValueError("root_quat contains zero quaternions")
        pose = torch.zeros(length, self.kinematics.num_bodies, 3)
        pose[:, 0] = torch.from_numpy(Rotation.from_quat(quat.numpy()).as_rotvec()).float()
        for index, name in enumerate(self.control_joint_names):
            body, axis = self.joints[name]
            pose[:, self.kinematics.body_names.index(body)] = q[:, index:index + 1] * torch.tensor(axis, dtype=torch.float32)
        fk = self.kinematics.fk_batch(pose[None], trans[None], dt=1 / fps)
        suffix = "_extend" if self.kinematics.cfg.extend_config else ""
        pos = getattr(fk, "global_translation" + suffix)[0]
        rot = getattr(fk, "global_rotation" + suffix)[0]
        vel = getattr(fk, "global_velocity" + suffix)[0]
        ang = getattr(fk, "global_angular_velocity" + suffix)[0]
        if dof_vel is None:
            dq = (q[1:] - q[:-1]) * fps
            dq = torch.cat((dq, dq[-1:]), dim=0) if length > 1 else torch.zeros_like(q)
        else:
            dq = _tensor(dof_vel, "dof_vel", q.shape)
        # Optional body derivatives are in this instance's body_names order.
        if body_vel is not None:
            vel = _tensor(body_vel, "body_vel", pos.shape)
        if body_ang_vel is not None:
            ang = _tensor(body_ang_vel, "body_ang_vel", pos.shape)
        if sample_times is not None:
            times = torch.as_tensor(sample_times, dtype=torch.float32).detach().cpu()
            if times.ndim != 1 or times.numel() == 0 or not torch.isfinite(times).all():
                raise ValueError("sample_times must be a nonempty finite vector")
            duration = (length - 1) / fps
            phase = (times / duration).clamp(0, 1) if duration else torch.zeros_like(times)
            i0 = (phase * (length - 1)).long()
            i1 = (i0 + 1).clamp(max=length - 1)
            blend = ((times.clamp_min(0) - i0 / fps) * fps).clamp(0, 1)
            def lerp(value):
                alpha = blend.reshape(-1, *([1] * (value.ndim - 1)))
                return (1 - alpha) * value[i0] + alpha * value[i1]
            q, dq, pos, vel, ang = [lerp(value) for value in (q, dq, pos, vel, ang)]
            rot = slerp(rot[i0], rot[i1], blend[:, None, None])
        result = compute_reference_observation(q, dq, pos, rot, vel, ang, self.default_dof_pos,
                                              root_height_obs=self.root_height_obs, obs_scales=self.obs_scales,
                                              device=self.device)
        if drop_first:
            if result["state"].shape[0] < 2:
                raise ValueError("drop_first needs at least two output frames")
            result = {key: value[1:] for key, value in result.items()}
        return result
