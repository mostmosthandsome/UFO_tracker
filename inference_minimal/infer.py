"""Standalone deterministic UFO actor inference; no UFO/simulator imports.

Network layout adapted from UFO ufo/agents/common/networks.py and inputs.py.
Copyright (c) Meta Platforms, Inc. and affiliates.
Distributed under CC BY-NC 4.0; see LICENSE in this directory.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation: bool):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim),
            *([nn.Mish()] if activation else []),
        )

    def forward(self, x):
        return self.mlp(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Mish())

    def forward(self, x):
        return x + self.mlp(x)


def embedding(input_dim: int, hidden_dim: int, layers: int):
    if layers < 2 or hidden_dim % 2:
        raise ValueError("Actor requires embedding_layers >= 2 and an even hidden_dim")
    return nn.Sequential(
        Block(input_dim, hidden_dim, True),
        *[ResidualBlock(hidden_dim) for _ in range(layers - 2)],
        Block(hidden_dim, hidden_dim // 2, True),
    )


class Actor(nn.Module):
    def __init__(self, obs_dim: int, z_dim: int, action_dim: int, config: dict):
        super().__init__()
        if config.get("name", "actor") != "actor":
            raise ValueError(f"Unsupported actor architecture: {config.get('name')}")
        width = config.get("hidden_dim", 1024)
        self.embed_z = embedding(obs_dim + z_dim, width, config.get("embedding_layers", 2))
        self.embed_s = embedding(obs_dim, width, config.get("embedding_layers", 2))
        self.policy = nn.Sequential(
            *[ResidualBlock(width) for _ in range(config.get("hidden_layers", 1))],
            Block(width, action_dim, False),
        )

    def forward(self, obs, z):
        features = torch.cat((self.embed_s(obs), self.embed_z(torch.cat((obs, z), dim=-1))), dim=-1)
        return torch.tanh(self.policy(features))


class BatchNormNormalizer(nn.Module):
    def __init__(self, dim: int, momentum: float = 0.01):
        super().__init__()
        self._normalizer = nn.BatchNorm1d(dim, affine=False, momentum=momentum)

    def forward(self, value):
        return self._normalizer(value)


class Norm(nn.Module):
    def forward(self, value):
        return math.sqrt(value.shape[-1]) * F.normalize(value, dim=-1)


class BackwardEncoder(nn.Module):
    def __init__(self, obs_dim: int, z_dim: int, config: dict):
        super().__init__()
        if config.get("name", "BackwardArchi") != "BackwardArchi":
            raise ValueError("Unsupported backward/goal encoder architecture")
        width = config.get("hidden_dim", 256)
        layers = [nn.Linear(obs_dim, width), nn.LayerNorm(width), nn.Tanh()]
        for _ in range(config.get("hidden_layers", 2) - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers += [nn.Linear(width, z_dim)]
        if config.get("norm", True):
            layers += [Norm()]
        self.net = nn.Sequential(*layers)

    def forward(self, value):
        return self.net(value)


def make_normalizer(config: dict, dim: int):
    if config["name"] == "IdentityNormalizerConfig":
        return nn.Identity()
    if config["name"] == "BatchNormNormalizerConfig":
        return BatchNormNormalizer(dim, config.get("momentum", 0.01))
    raise ValueError(f"Unsupported observation normalizer: {config['name']}")


def vector_dim(space: dict) -> int:
    if space.get("type") != "Box" or len(space["shape"]) != 1:
        raise ValueError("Only one-dimensional Box observation fields are supported")
    return int(space["shape"][0])


def resolve_model_dir(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if (path / "checkpoint").is_dir():
        path = path / "checkpoint"
    if (path / ".saving").exists() or (path.name == "model" and (path.parent / ".saving").exists()):
        raise RuntimeError("Checkpoint save is in progress; use a completed snapshot")
    if (path / "model").is_dir():
        path = path / "model"
    for name in ("config.json", "init_kwargs.json", "model.safetensors"):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    return path


class Policy(nn.Module):
    """Load once, then call policy(observation_dict, z) for deterministic actions."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu", *,
                 reference_metadata: dict | None = None, reference_xml: str | Path | None = None):
        super().__init__()
        self.reference_metadata = reference_metadata
        self.reference_xml = reference_xml
        self._reference_computer = None
        self.model_dir = resolve_model_dir(checkpoint)
        config = json.loads((self.model_dir / "config.json").read_text())
        kwargs = json.loads((self.model_dir / "init_kwargs.json").read_text())
        if config["name"] not in ("FBcprAuxModel", "TeCHModel"):
            raise ValueError(f"Unsupported UFO model: {config['name']}")
        self.device = torch.device(device)
        arch = config["archi"]
        self.model_name = config["name"]
        self.arch = arch
        self._encoder = None
        self.z_dim = int(arch["z_dim"])
        self.action_dim = int(kwargs["action_dim"])
        space = kwargs["obs_space"]
        self.is_dict = space["type"] == "Dict"
        self.obs_dims = ({key: vector_dim(value) for key, value in space["spaces"].items()}
                         if self.is_dict else {"observation": vector_dim(space)})
        filter_config = arch["actor"].get("input_filter", {"name": "IdentityInputFilterConfig"})
        if filter_config["name"] == "DictInputFilterConfig" and self.is_dict:
            keys = filter_config["key"]
            self.input_keys = [keys] if isinstance(keys, str) else list(keys)
        elif filter_config["name"] == "IdentityInputFilterConfig" and not self.is_dict:
            self.input_keys = ["observation"]
        else:
            raise ValueError("Unsupported actor observation filter/space combination")
        obs_dim = sum(self.obs_dims[key] for key in self.input_keys)
        self._actor = Actor(obs_dim, self.z_dim, self.action_dim, arch["actor"])
        normalization = config["obs_normalizer"]["normalizers"]
        self._obs_normalizer = nn.Module()
        if self.is_dict:
            if "name" in normalization:
                raise ValueError("Dict observations require per-field normalizer configuration")
            self._obs_normalizer._normalizers = nn.ModuleDict({
                key: make_normalizer(value, self.obs_dims[key]) for key, value in normalization.items()
            })
            missing = set(self.input_keys) - set(normalization)
            if missing:
                raise ValueError(f"Actor input fields missing from normalizer: {sorted(missing)}")
        else:
            self._obs_normalizer._normalizers = make_normalizer(normalization, obs_dim)
        # Read only inference tensors, even if the file contains large critics/targets.
        prefixes = ("_actor.", "_obs_normalizer.")
        with safe_open(self.model_dir / "model.safetensors", framework="pt", device="cpu") as saved:
            state = {key: saved.get_tensor(key) for key in saved.keys() if key.startswith(prefixes)}
        self.load_state_dict(state, strict=True)
        self.to(self.device).eval().requires_grad_(False)

    def _load_encoder(self):
        if self._encoder is not None:
            return
        is_fb = self.model_name == "FBcprAuxModel"
        config = self.arch["b" if is_fb else "goal_encoder"]
        filter_config = config.get("input_filter", {"name": "IdentityInputFilterConfig"})
        if filter_config["name"] == "DictInputFilterConfig" and self.is_dict:
            keys = filter_config["key"]
            self.encoder_input_keys = [keys] if isinstance(keys, str) else list(keys)
        elif filter_config["name"] == "IdentityInputFilterConfig" and not self.is_dict:
            self.encoder_input_keys = ["observation"]
        else:
            raise ValueError("Unsupported encoder observation filter/space combination")
        encoder = BackwardEncoder(sum(self.obs_dims[key] for key in self.encoder_input_keys), self.z_dim, config)
        prefix = "_backward_map." if is_fb else "_goal_encoder."
        with safe_open(self.model_dir / "model.safetensors", framework="pt", device="cpu") as saved:
            state = {key[len(prefix):]: saved.get_tensor(key) for key in saved.keys() if key.startswith(prefix)}
        encoder.load_state_dict(state, strict=True)
        self._encoder = encoder.to(self.device).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self, ref_action, *, root_pos=None, root_quat=None, fps=None,
               joint_names=None, quat_order="xyzw", sample_times=None, drop_first=False):
        """G1 joint angles [L,29] plus root trajectory/FPS -> latents [N,z_dim].

        Raw G1 inputs compute reference features via UFO_observation internally.
        Existing feature dictionaries (and flat checkpoints' feature arrays) remain
        supported. No implicit averaging or next-frame shift is applied.
        """
        if self.training:
            raise RuntimeError("Policy is inference-only; keep it in eval() mode")
        if self.is_dict and not isinstance(ref_action, dict):
            if root_pos is None or root_quat is None or fps is None:
                raise ValueError("Raw G1 joint angles require root_pos [L,3], root_quat [L,4], and fps")
            if self.action_dim != 29:
                raise ValueError("Raw reference encoding is configured for G1 (29 joints)")
            if self._reference_computer is None:
                if __package__:
                    from .UFO_observation import make_g1_computer
                else:
                    from UFO_observation import make_g1_computer
                self._reference_computer = make_g1_computer(
                    device=self.device, xml_path=self.reference_xml, metadata=self.reference_metadata,
                )
            ref_action = self._reference_computer.from_robot_state(
                ref_action, root_pos, root_quat, fps, joint_names=joint_names,
                quat_order=quat_order, sample_times=sample_times, drop_first=drop_first,
            )
        elif any(value is not None for value in (root_pos, root_quat, fps, joint_names, sample_times)) or drop_first or quat_order != "xyzw":
            raise ValueError("Root trajectory/sampling arguments apply only to raw G1 joint-angle inputs")
        self._load_encoder()
        values = []
        batch_size = None
        for key in self.encoder_input_keys:
            if self.is_dict and key not in ref_action:
                raise ValueError(f"Encoder requires reference field {key!r}, which was not supplied/computed")
            value = torch.as_tensor(ref_action[key] if self.is_dict else ref_action,
                                    dtype=torch.float32, device=self.device)
            if value.ndim != 2 or value.shape[1] != self.obs_dims[key] or value.shape[0] < 1:
                raise ValueError(f"Reference {key} must have shape [N, {self.obs_dims[key]}], N >= 1")
            if batch_size is not None and value.shape[0] != batch_size:
                raise ValueError("Reference observation fields must have the same number of rows")
            batch_size = value.shape[0]
            norm = self._obs_normalizer._normalizers[key] if self.is_dict else self._obs_normalizer._normalizers
            values.append(norm(value))
        z = self._encoder(torch.cat(values, dim=-1))
        if self.model_name == "TeCHModel" and self.arch.get("z_geometry", "hypersphere") == "poincare_ball":
            radius = float(self.arch.get("poincare_radius", 1.0))
            eps = float(self.arch.get("poincare_eps", 1e-6))
            norm = torch.linalg.norm(z, dim=-1, keepdim=True).clamp_min(eps)
            return radius * torch.tanh(norm / radius) * (z / norm)
        if self.arch.get("norm_z", True):
            z = math.sqrt(self.z_dim) * F.normalize(z, dim=-1)
        return z

    def zero_inputs(self, batch_size: int = 1):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        obs = {key: torch.zeros(batch_size, self.obs_dims[key], device=self.device) for key in self.input_keys}
        if not self.is_dict:
            obs = obs["observation"]
        return obs, torch.zeros(batch_size, self.z_dim, device=self.device)

    @torch.inference_mode()
    def forward(self, obs, z):
        if self.training:
            raise RuntimeError("Policy is inference-only; keep it in eval() mode")
        z = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        if z.ndim != 2 or z.shape[1] != self.z_dim:
            raise ValueError(f"z must have shape [batch, {self.z_dim}]")
        values = []
        for key in self.input_keys:
            value = torch.as_tensor(obs[key] if self.is_dict else obs, dtype=torch.float32, device=self.device)
            if value.shape != (z.shape[0], self.obs_dims[key]):
                raise ValueError(f"{key} must have shape [{z.shape[0]}, {self.obs_dims[key]}]")
            norm = self._obs_normalizer._normalizers[key] if self.is_dict else self._obs_normalizer._normalizers
            values.append(norm(value))
        return self._actor(torch.cat(values, dim=-1), z).float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Run directory, checkpoint directory, or model directory")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--encode-reference", action="store_true", help="Encode zero reference observations into z before actor inference")
    args = parser.parse_args()
    policy = Policy(args.checkpoint, device=args.device)
    obs, z = policy.zero_inputs(args.batch_size)
    if args.encode_reference:
        reference = {key: torch.zeros(args.batch_size, dim, device=policy.device) for key, dim in policy.obs_dims.items()}
        z = policy.encode(reference if policy.is_dict else reference["observation"])
        print("Encoder input order:", policy.encoder_input_keys)
    action = policy(obs, z)
    print("Model:", policy.model_dir)
    print("Actor input order:", policy.input_keys)
    print("Observation dimensions:", {key: policy.obs_dims[key] for key in policy.input_keys})
    print("z shape:", tuple(z.shape))
    if args.encode_reference:
        print("z:", z.cpu().tolist())
    print("Action shape:", tuple(action.shape))
    print("Action:", action.cpu().tolist())


if __name__ == "__main__":
    main()
