"""Integration of raw G1 references, observation preprocessing and B inference."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from inference_minimal.infer import Actor, BackwardEncoder, Policy
from inference_minimal.UFO_observation import make_g1_computer


class G1EncodeTest(unittest.TestCase):
    def test_raw_reference_matches_feature_encoding(self):
        torch.manual_seed(19)
        computer = make_g1_computer()
        q = torch.randn(8, 29) * 0.1
        pos = torch.zeros(8, 3)
        pos[:, 2] = 0.8
        quat = torch.tensor([0., 0., 0., 1.]).repeat(8, 1)
        features = computer.from_robot_state(q, pos, quat, 30)
        dims = {key: value.shape[1] for key, value in features.items()}
        actor = dict(hidden_dim=16, hidden_layers=1, embedding_layers=2,
                     input_filter=dict(name="DictInputFilterConfig", key=["state"]))
        backward = dict(hidden_dim=16, hidden_layers=2, norm=True,
                        input_filter=dict(name="DictInputFilterConfig", key=["state", "privileged_state"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps(dict(
                name="FBcprAuxModel", archi=dict(z_dim=8, actor=actor, b=backward),
                obs_normalizer=dict(normalizers={key: dict(name="IdentityNormalizerConfig") for key in dims}),
            )))
            (path / "init_kwargs.json").write_text(json.dumps(dict(
                action_dim=29, obs_space=dict(type="Dict", spaces={
                    key: dict(type="Box", shape=[dim]) for key, dim in dims.items()}),
            )))
            weights = {"_actor."+k: v for k, v in Actor(64, 8, 29, actor).state_dict().items()}
            weights.update({"_backward_map."+k: v for k, v in
                            BackwardEncoder(64+dims["privileged_state"], 8, backward).state_dict().items()})
            save_file(weights, path / "model.safetensors")
            policy = Policy(path)
            expected = policy.encode(features)
            actual = policy.encode(q, root_pos=pos, root_quat=quat, fps=30)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertEqual(actual.shape, (8, 8))
            reordered = policy.encode(q.flip(1), root_pos=pos, root_quat=quat, fps=30,
                                      joint_names=computer.control_joint_names[::-1])
            torch.testing.assert_close(reordered, expected, atol=0, rtol=0)
            torch.testing.assert_close(policy.encode(q, root_pos=pos, root_quat=quat, fps=30,
                                                     drop_first=True), expected[1:])
            with self.assertRaisesRegex(ValueError, "require root_pos"):
                policy.encode(q)
            with self.assertRaises(ValueError):
                policy.encode(q[:, :28], root_pos=pos, root_quat=quat, fps=30)
            obs, _ = policy.zero_inputs(8)
            self.assertTrue(torch.isfinite(policy(obs, actual)).all())


if __name__ == "__main__":
    unittest.main()
