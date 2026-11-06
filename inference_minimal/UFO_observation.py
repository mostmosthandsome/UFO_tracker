"""G1 reference trajectories -> UFO observations, without a simulator."""
import json
from pathlib import Path

if __package__:
    from .UFO_observation_compute import ReferenceObservationComputer
else:
    from UFO_observation_compute import ReferenceObservationComputer


def make_g1_computer(*, device="cpu", xml_path=None, metadata=None):
    """Use bundled repository defaults, or an explicit training metadata dict."""
    assets = Path(__file__).resolve().parent / "UFO_observation_compute"
    if metadata is None:
        metadata = json.loads((assets / "g1_metadata.json").read_text())
    if len(metadata["control_joint_names"]) != 29:
        raise ValueError("G1 reference encoding requires 29 control joints")
    return ReferenceObservationComputer(
        xml_path if xml_path is not None else assets / "g1_29dof.xml",
        **metadata, device=device,
    )
