"""Import the 23-DOF DR02 assets/configuration from whole_body_tracking.

Run from the UFO repository root. Isaac Lab is not needed: configuration values
are read with ast.literal_eval, never imported or executed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import yaml

from ufo.tools.robot_config import load_robot_training_spec


def keywords(call):
    return {kw.arg: kw.value for kw in call.keywords}


def import_dr02(source: Path, *, force: bool = False):
    package = source / "source/whole_body_tracking/whole_body_tracking"
    urdf = package / "assets/dr02_description/dr02_pro.urdf"
    config_source = package / "robots/dr02.py"
    assignments = {node.targets[0].id: node.value for node in ast.parse(config_source.read_text()).body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    cfg = keywords(assignments["DR02_robot_cfg"])
    initial = {key: ast.literal_eval(value) for key, value in keywords(cfg["init_state"]).items()}
    action_scales = ast.literal_eval(assignments["DR02_ACTION_SCALE"])
    if set(action_scales.values()) != {0.25} or len(action_scales) != 23:
        raise ValueError("Expected the source 23-DOF DR02 configuration with uniform 0.25 action scale")
    joint_order = list(action_scales)
    groups = [keywords(value) for value in cfg["actuators"].values]
    joint_params = {}
    for name in joint_order:
        matched = [group for group in groups if any(re.fullmatch(pattern, name)
                   for pattern in ast.literal_eval(group["joint_names_expr"]))]
        if len(matched) != 1:
            raise ValueError(f"Expected exactly one actuator group for {name}")
        params = {}
        for key in ("stiffness", "damping", "effort_limit_sim", "velocity_limit_sim", "armature"):
            value = ast.literal_eval(matched[0][key])
            if isinstance(value, dict):
                values = [v for pattern, v in value.items() if re.fullmatch(pattern, name)]
                if len(values) != 1:
                    raise ValueError(f"Ambiguous {key} for {name}")
                value = values[0]
            params[key] = float(value)
        joint_params[name] = params

    output = Path("ufo/data/robots/dr02")
    config_path = Path("configs/robots/dr02_23dof.yaml")
    xml_path = output / "dr02_23dof.xml"
    if not force and (xml_path.exists() or config_path.exists()):
        raise FileExistsError("DR02 already exists; use --force to regenerate")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(urdf, output / urdf.name)
    tree = ET.parse(urdf)
    for mesh in tree.findall(".//mesh"):
        relative = Path(mesh.get("filename"))
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(urdf.parent / relative, target)
        mesh.set("filename", relative.name)
    extension = ET.SubElement(tree.getroot(), "mujoco")
    ET.SubElement(extension, "compiler", meshdir=str((output / "meshes").resolve()),
                  discardvisual="false", fusestatic="false")
    with tempfile.TemporaryDirectory() as temp:
        converted = Path(temp) / "dr02.urdf"
        tree.write(converted)
        model = mujoco.MjModel.from_xml_path(str(converted))
        mujoco.mj_saveLastXML(str(xml_path), model)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    root.set("model", "dr02_23dof")
    root.find("compiler").set("meshdir", "meshes")
    base = root.find("worldbody/body")
    if base.get("name") != "base_link":
        raise ValueError("Expected base_link root")
    base.set("pos", " ".join(map(str, initial["pos"])))
    ET.SubElement(base, "freejoint", name="root")
    ET.SubElement(root, "option", timestep="0.005", integrator="implicitfast")
    # Follow the model traversal, which is also the UFO FK joint order.
    joint_order = [joint.get("name") for joint in root.findall(".//body/joint")]
    if set(joint_order) != set(action_scales):
        raise ValueError("URDF joints differ from DR02 robot configuration")
    actuators = ET.SubElement(root, "actuator")
    for name in joint_order:
        effort = joint_params[name]["effort_limit_sim"]
        ET.SubElement(actuators, "motor", name=name, joint=name, ctrllimited="true",
                      ctrlrange=f"{-effort} {effort}")
    # mjlab owns motor limits; it explicitly rejects joint actuatorfrcrange.
    for joint in root.findall(".//body/joint"):
        joint.attrib.pop("actuatorfrcrange", None)
    for geom in root.findall(".//geom"):
        # Match source UrdfFileCfg.replace_cylinders_with_capsules.
        if geom.get("type") == "cylinder":
            geom.set("type", "capsule")
        visual = geom.get("contype") == "0" and geom.get("conaffinity") == "0"
        geom.set("group", "2" if visual else "3")
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    mujoco.MjModel.from_xml_path(str(xml_path))
    feet = [f"{side}_ankle_x_link" for side in ("left", "right")]
    hands = [f"{side}_wrist_x_link" for side in ("left", "right")]
    config = {
        "name": "dr02_23dof", "xml_path": str(xml_path), "base_body": "base_link",
        "root_quat_order": "xyzw", "coordinate_system": "z_up", "dof_unit": "rad",
        "control_joints": joint_order, "feet": feet, "hands": hands,
        "key_bodies": ["base_link", "body", *feet, *hands], "default_dof_pos": {},
        "training": {
            "motion_extensions": [],
            "task_overrides": {"rewards": {"reward_limit": {"soft_dof_pos_limit": 0.9}}},
            "semantics": {"contact_bodies": feet,
                          "undesired_contact_bodies": ["base_link", "body", "hip", "knee", "shoulder", "elbow"],
                          "torso_name": "body",
                          "left_ankle_dof_names": ["left_ankle_y_joint", "left_ankle_x_joint"],
                          "right_ankle_dof_names": ["right_ankle_y_joint", "right_ankle_x_joint"]},
            "init_state": {"pos": list(initial["pos"]),
                           "rot": [*initial["rot"][1:], initial["rot"][0]],
                           "lin_vel": [0.0] * 3, "ang_vel": [0.0] * 3,
                           "default_joint_angles": initial["joint_pos"]},
            "control": {"action_scale": 0.25, "action_rescale": False,
                        "action_clip_value": 1.0, "normalize_action_to": 1.0, "effort_limit_scale": 1.0,
                        "effort_limit": [joint_params[n]["effort_limit_sim"] for n in joint_order],
                        "velocity_limit": [joint_params[n]["velocity_limit_sim"] for n in joint_order],
                        "stiffness": {n: joint_params[n]["stiffness"] for n in joint_order},
                        "damping": {n: joint_params[n]["damping"] for n in joint_order}},
            "actuator": {"source": "yaml", "model": "dc_motor", "joints": {
                n: {"effort_limit": joint_params[n]["effort_limit_sim"],
                    "velocity_limit": joint_params[n]["velocity_limit_sim"],
                    "armature": joint_params[n]["armature"], "friction": 0.0} for n in joint_order}},
        },
        "metadata": {"source": "whole_body_tracking: robots/dr02.py + dr02_pro.urdf (23 DOF)",
                     "source_urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
                     "source_config_sha256": hashlib.sha256(config_source.read_bytes()).hexdigest(),
                     "review_status": "simulation_port",
                     "notes": ["Source implicit PD is approximated with UFO dc_motor; dynamics are not identical.",
                               "UFO task rewards/domain randomization apply, not source PPO tracking rewards.",
                               "Fixed wrist_x bodies represent hands; this model has no actuated wrists."]},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    spec = load_robot_training_spec(config_path)
    print(f"Imported {spec.robot.name}: {len(spec.robot.control_joint_names)} joints -> {config_path}")


def import_motions(source: Path, paths: list[Path], fps: float, *, force: bool = False):
    """Name source CSV columns before UFO reorders them into model order."""
    from ufo.motion.formats import RobotStateMotion, validate_robot_state_motion
    from ufo.tools.motion_data import main as motion_main

    calls = [node for node in ast.walk(ast.parse((source / "scripts/csv_to_npz.py").read_text()))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_simulator"]
    orders = [ast.literal_eval(keywords(call)["joint_names"]) for call in calls if "joint_names" in keywords(call)]
    if len(orders) != 1:
        raise ValueError("Cannot determine unambiguous source CSV joint order")
    names = orders[0]
    spec = load_robot_training_spec("configs/robots/dr02_23dof.yaml").robot
    output = Path("cache/dr02_robot_state")
    manifest = Path("configs/data/dr02_local.yaml")
    targets = [output / (path.stem + ".npz") for path in paths]
    if len(set(targets)) != len(targets):
        raise ValueError("Motion filenames must have unique stems")
    if not force and any(path.exists() for path in [manifest, *targets]):
        raise FileExistsError("Motion outputs already exist; use --force to regenerate")
    motions = []
    for path in paths:
        values = np.loadtxt(path, delimiter=",", ndmin=2)
        if values.shape[1] != 7 + len(names):
            raise ValueError(f"{path}: expected root XYZ + XYZW + {len(names)} joints")
        motion = RobotStateMotion(motion_key=path.stem, root_pos=values[:, :3], root_quat=values[:, 3:7],
                                  dof_pos=values[:, 7:], fps=fps, joint_names=names)
        motions.append(validate_robot_state_motion(motion, spec, str(path)))
    output.mkdir(parents=True, exist_ok=True)
    for target, motion in zip(targets, motions):
        np.savez_compressed(target, root_pos=motion.root_pos, root_quat=motion.root_quat,
                            dof_pos=motion.dof_pos, joint_names=spec.control_joint_names, fps=fps)
    args = ["build", "--robot", "configs/robots/dr02_23dof.yaml", "--source", *map(str, targets),
            "--format", "robot_state_npz", "--name", "dr02_local", "--out", str(manifest)]
    if force:
        args += ["--force", "--rebuild-cache"]
    motion_main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="whole_body_tracking repository")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--motion-source", nargs="+", type=Path, help="Source DR02 headerless CSV files")
    parser.add_argument("--fps", type=float, help="Actual CSV sample rate (required with --motion-source)")
    args = parser.parse_args()
    if args.motion_source and (args.fps is None or not np.isfinite(args.fps) or args.fps <= 0):
        parser.error("--motion-source requires a finite positive --fps")
    import_dr02(args.source.resolve(), force=args.force)
    if args.motion_source:
        import_motions(args.source.resolve(), args.motion_source, args.fps, force=args.force)
