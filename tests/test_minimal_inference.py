"""Development checks against UFO; the standalone runtime does not import UFO."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import safetensors.torch
import torch
from torch import nn

from inference_minimal.infer import Policy
from ufo.agents.common.checkpoint import save_build_kwargs
from ufo.agents.common.inputs import (
    BatchNormNormalizerConfig, DictInputFilterConfig, IdentityInputFilterConfig,
    IdentityNormalizerConfig, ObsNormalizerConfig,
)
from ufo.agents.common.networks import ActorArchiConfig


class MinimalInferenceTest(unittest.TestCase):
    def test_matches_ufo_actor_and_normalization(self):
        torch.manual_seed(17)
        for model_name, mode in [("FBcprAuxModel", "dict"), ("TeCHModel", "single"), ("FBcprAuxModel", "flat")]:
            with self.subTest(model=model_name, mode=mode), tempfile.TemporaryDirectory() as temp:
                space = gym.spaces.Box(-1.0, 1.0, shape=(6,))
                if mode != "flat":
                    space = gym.spaces.Dict({
                        "state": gym.spaces.Box(-1.0, 1.0, shape=(6,)),
                        "last_action": gym.spaces.Box(-1.0, 1.0, shape=(3,)),
                        "privileged_state": gym.spaces.Box(-1.0, 1.0, shape=(4,)),
                    })
                actor_config = ActorArchiConfig(
                    hidden_dim=16, hidden_layers=2, embedding_layers=3,
                    input_filter=(IdentityInputFilterConfig() if mode == "flat" else
                                  DictInputFilterConfig(key="state" if mode == "single" else ["last_action", "state"])),
                )
                norm_config = ObsNormalizerConfig(normalizers=(BatchNormNormalizerConfig() if mode == "flat" else {
                    "state": BatchNormNormalizerConfig(), "last_action": IdentityNormalizerConfig(),
                    "privileged_state": BatchNormNormalizerConfig(),
                }))
                reference = nn.Module()
                reference._actor = actor_config.build(space, 8, 3)
                reference._obs_normalizer = norm_config.build(space)
                # Nontrivial saved statistics catch accidentally bypassed normalization.
                for module in reference.modules():
                    if isinstance(module, nn.BatchNorm1d):
                        module.running_mean.copy_(torch.randn_like(module.running_mean))
                        module.running_var.copy_(torch.rand_like(module.running_var) + 0.2)
                reference.eval()
                root = Path(temp)
                model = root / "checkpoint/model"
                model.mkdir(parents=True)
                save_build_kwargs(model, {"obs_space": space, "action_dim": 3})
                (model / "config.json").write_text(json.dumps({
                    "name": model_name, "archi": {"z_dim": 8, "actor": actor_config.model_dump()},
                    "obs_normalizer": norm_config.model_dump(),
                }))
                weights = reference.state_dict()
                weights["_critic.unused"] = torch.ones(1)
                weights["_target_actor.unused"] = torch.ones(1)
                safetensors.torch.save_file(weights, model / "model.safetensors")
                policy = Policy(root)
                for batch in (1, 4):
                    obs = (torch.randn(batch, 6) if mode == "flat" else
                           {key: torch.randn(batch, *value.shape) for key, value in space.spaces.items()})
                    z = torch.randn(batch, 8)
                    with torch.no_grad():
                        expected = reference._actor(reference._obs_normalizer(obs), z, 0.2).mean
                    torch.testing.assert_close(policy(obs, z), expected, rtol=0, atol=0)
                    zero_obs, zero_z = policy.zero_inputs(batch)
                    self.assertTrue(torch.isfinite(policy(zero_obs, zero_z)).all())
                with self.assertRaises(ValueError):
                    policy(obs, torch.zeros(4, 7))
                # The copied script runs outside the repository with isolated sys.path.
                script = root / "infer.py"
                shutil.copyfile(Path(__file__).resolve().parents[1] / "inference_minimal/infer.py", script)
                result = subprocess.run(
                    [sys.executable, "-I", str(script), "--checkpoint", str(model)],
                    cwd=root, capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Action shape: (1, 3)", result.stdout)


if __name__ == "__main__":
    unittest.main()
