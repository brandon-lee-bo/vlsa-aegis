"""Build or restore one table-mounted cylinder on a recorded Level I trajectory."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from xml.etree.ElementTree import SubElement

import mujoco
import numpy as np

from .level_three import _bounds

NAME = "paired_path_obstacle"
RADIUS = .024
HEIGHTS = (.22, .32, .40)
CATALOG = Path(__file__).with_name("paired_level_ii.json")


def add_blocker(model):
    body = SubElement(model.worldbody, "body", name=NAME, mocap="true", pos="0 0 -8")
    SubElement(body, "geom", type="cylinder", name=NAME, size=f"{RADIUS} {max(HEIGHTS)/2}",
               rgba=".93 .35 .08 1", group="1", contype="1", conaffinity="1")


def state_key(task_file, state):
    digest = hashlib.sha256(np.asarray(state, dtype="<f8").tobytes()).hexdigest()
    return f"{Path(task_file).stem}/{digest}"


@lru_cache(maxsize=1)
def frozen_cases():
    return json.loads(CATALOG.read_text())["cases"]


def restore_case(env, state, case=None):
    key = state_key(env.bddl_file_name, state)
    if case is None:
        case = frozen_cases().get(key)
    if case is None or case["key"] != key:
        raise ValueError(f"No verified paired obstacle for {key}. Generate a case from its Level I rollout.")
    sim = env.sim
    mid = sim.model._model.body_mocapid[sim.model.body_name2id(NAME)]
    geom = sim.model.geom_name2id(NAME)
    set_height(sim.model._model,geom,case.get("height_m",HEIGHTS[0]))
    sim.data._data.mocap_pos[mid] = case["center"]


def set_height(model, geom, height):
    # Compile bounds at the largest size. Changing the analytic cylinder inside
    # those conservative bounds keeps both rendering and narrow-phase accurate.
    if height not in HEIGHTS:
        raise ValueError(f"Unsupported cylinder height {height}")
    model.geom_size[geom,1] = height/2


def robot_geoms(env):
    sim = env.sim
    model = sim.model._model
    root = sim.model.body_name2id(env.robots[0].robot_model.root_body)
    geoms = []
    for i, body in enumerate(model.geom_bodyid):
        while body and body != root:
            body = model.body_parentid[body]
        if body == root and (model.geom_contype[i] or model.geom_conaffinity[i]):
            geoms.append(i)
    return geoms


def build_case(env, state, poses, gripper):
    """Restore the reference start, then test recorded poses in a separate MuJoCo data object."""
    env.set_state(state)
    env.sim.forward()
    sim = env.sim
    model, live = sim.model._model, sim.data._data
    block = sim.model.geom_name2id(NAME)
    mid = model.body_mocapid[sim.model.body_name2id(NAME)]
    data = mujoco.MjData(model)
    data.qpos[:] = live.qpos
    data.mocap_pos[:] = live.mocap_pos
    data.mocap_quat[:] = live.mocap_quat
    mujoco.mj_forward(model, data)
    boxes = {i: _bounds(model, data, i) for i in range(model.ngeom)
             if i != block and (model.geom_contype[i] or model.geom_conaffinity[i])}
    surfaces = [i for i,(a,b) in boxes.items() if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
                and min((b-a)[:2]) >= .5 and abs(b[2]-env.env.workspace_offset[2]) < .09]
    if env.env.workspace_name == "floor":
        support = sim.model.geom_name2id("floor")
        low = np.array([-.46,-.46,0.])
        high = np.array([.46,.46,data.geom_xpos[support,2]])
    elif surfaces:
        support = max(surfaces, key=lambda i: boxes[i][1][2])
        low, high = boxes[support]
    else:
        raise ValueError("No workspace support for the added obstacle")
    robots = set(robot_geoms(env))
    poses, gripper = np.asarray(poses), np.asarray(gripper)
    if poses.shape != (len(gripper), model.nq) or not len(poses):
        raise ValueError("Expected one complete simulator qpos per gripper sample")
    # Prefer an early approach, after the arm has moved five centimetres.
    order = [i for i in range(1,len(poses)) if np.linalg.norm(gripper[i]-gripper[0]) >= .05]
    offsets = [(0,0), (.025,0), (-.025,0), (0,.025), (0,-.025),
               (.045,0), (-.045,0), (0,.045), (0,-.045)]
    at_reset = mujoco.MjData(model)
    at_reset.qpos[:] = live.qpos
    at_reset.mocap_pos[:] = live.mocap_pos
    at_reset.mocap_quat[:] = live.mocap_quat

    def scan(anchor, poses_to_check, surface, exact_clearance, height):
        set_height(model,block,height)
        limits = np.array([RADIUS,RADIUS,height/2])
        floor, ceiling = boxes[surface] if surface != support else (low, high)
        for i in poses_to_check:
            for offset in offsets:
                center = np.r_[anchor[i,:2]+offset, ceiling[2]+height/2]
                a, b = center-limits, center+limits
                if (a[:2] < floor[:2]).any() or (b[:2] > ceiling[:2]).any():
                    continue
                nearby = [j for j,(lo,hi) in boxes.items() if j != surface
                          and np.all(hi+.002 > a) and np.all(lo-.002 < b)]
                if nearby:
                    if not exact_clearance:
                        continue
                    at_reset.mocap_pos[mid] = center
                    mujoco.mj_forward(model,at_reset)
                    if any(mujoco.mj_geomDistance(model,at_reset,block,j,.003,None) < .002
                           for j in nearby):
                        continue
                data.mocap_pos[mid] = center
                data.qpos[:] = poses[i]
                mujoco.mj_forward(model,data)
                for contact in data.contact:
                    g1,g2 = int(contact.geom1),int(contact.geom2)
                    if block not in (g1,g2) or contact.dist >= -.001:
                        continue
                    other = g2 if g1 == block else g1
                    if other in robots:
                        return dict(key=state_key(env.bddl_file_name,state),center=center.tolist(),
                                    height_m=height,reference_step=int(i),geom=sim.model.geom_id2name(other),
                                    penetration_m=float(-contact.dist),contact=contact.pos.tolist())
        return None

    case = scan(gripper,order,support,False,HEIGHTS[0])
    if case:
        return case
    # The reference may reach onto a raised cabinet. Include horizontal, fixed
    # work surfaces and use their actual MuJoCo geometry for clearance.
    fixed_tops = [j for j,(a,b) in boxes.items() if j != support
                  and model.geom_type[j] == mujoco.mjtGeom.mjGEOM_BOX
                  and model.body_weldid[model.geom_bodyid[j]] == 0
                  and abs(data.geom_xmat[j].reshape(3,3)[2,2]) > .999
                  and min((b-a)[:2]) >= 2*RADIUS
                  and high[2]+.01 < b[2] < high[2]+.55]
    for height in HEIGHTS:
        for surface in [support]+fixed_tops:
            case = scan(gripper,order,surface,True,height)
            if case:
                return case
    raise ValueError("No initially clear, supported placement intersects this reference trajectory")


def start_contact_monitor(env):
    """Count original and added obstacle contacts from this point onward."""
    sim = env.sim
    sim._paired_first_added_sim_time = None
    if hasattr(sim, "_paired_contact_flags"):
        sim._paired_contact_flags.update(original=False, added=False)
        return sim._paired_contact_flags
    model, data = sim.model._model, sim.data._data
    robots = set(robot_geoms(env))
    roots = {v for k,v in env.env.obj_body_id.items() if "obstacle" in k}
    hazards = set()
    for g, body in enumerate(model.geom_bodyid):
        while body and body not in roots:
            body = model.body_parentid[body]
        if body in roots:
            hazards.add(g)
    added = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,NAME)
    if added >= 0:
        hazards.add(added)
    flags = {"original":False, "added":False}
    advance = sim.step
    def step():
        substep_start = float(data.time)
        result = advance()
        for c in data.contact:
            if c.dist > 0:
                continue
            a,b = int(c.geom1),int(c.geom2)
            if b in robots:
                a,b = b,a
            if a in robots and b in hazards:
                if b == added and not flags["added"]:
                    sim._paired_first_added_sim_time = substep_start
                flags["added" if b == added else "original"] = True
        return result
    sim._paired_contact_flags = flags
    sim.step = step
    return flags
