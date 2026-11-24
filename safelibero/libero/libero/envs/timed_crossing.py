"""Time a supported straight crossing against a recorded 20 Hz Level I rollout."""

import json
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from .level_three import _bounds
from .paired_obstacle import HEIGHTS, NAME, RADIUS, robot_geoms, set_height

SPEEDS = (0.05, 0.10, 0.20, 0.40, 0.80)
DT = 0.05


@lru_cache(maxsize=1)
def timed_cases():
    return json.loads(Path(__file__).with_name("timed_level_iii.json").read_text())["cases"]


class TimedCrossing:
    def __init__(self, sim, case, speed, policy_start):
        if not np.isfinite(speed) or speed < 0:
            raise ValueError("Expected a finite nonnegative obstacle speed")
        self.start = np.asarray(case["start"])
        self.end = np.asarray(case["end"])
        self.direction = self.end - self.start
        self.length = np.linalg.norm(self.direction)
        self.direction = self.direction / self.length
        self.speed = speed
        self.arrival = case["step"] * DT
        self.onset = self.arrival - case["lead_m"] / speed if speed else np.inf
        if self.onset < -1e-8:
            raise ValueError("This speed cannot reach the crossing from the frozen start in time")
        self.policy_start = policy_start
        self.mid = sim.model._model.body_mocapid[sim.model.body_name2id(NAME)]
        set_height(sim.model._model, sim.model.geom_name2id(NAME), case["height_m"])

    def position(self, policy_time):
        travel = (
            np.clip((policy_time - self.onset) * self.speed, 0, self.length) if self.speed else 0.0
        )
        return self.start + self.direction * travel

    def apply(self, sim):
        sim.data._data.mocap_pos[self.mid] = self.position(sim.data.time - self.policy_start)
        sim.forward()


def lane_checker(env, state, fixed):
    """Return a swept-clearance check against the original reset geometry."""
    env.set_state(state)
    sim = env.sim
    sim.forward()
    model, initial = sim.model._model, sim.data._data
    block = sim.model.geom_name2id(NAME)
    mid = model.body_mocapid[sim.model.body_name2id(NAME)]
    height = fixed.get("height_m", HEIGHTS[0])
    set_height(model, block, height)
    data = mujoco.MjData(model)
    boxes = {
        j: _bounds(model, initial, j)
        for j in range(model.ngeom)
        if j != block and (model.geom_contype[j] or model.geom_conaffinity[j])
    }
    bottom = fixed["center"][2] - height / 2
    if env.env.workspace_name == "floor":
        support = sim.model.geom_name2id("floor")
        low, high = np.array([-0.46, -0.46]), np.array([0.46, 0.46])
    else:
        surfaces = [
            j
            for j, (a, b) in boxes.items()
            if abs(b[2] - bottom) < 0.002
            and model.geom_type[j] == mujoco.mjtGeom.mjGEOM_BOX
            and min((b - a)[:2]) > 2 * RADIUS
        ]
        support = max(surfaces, key=lambda j: np.prod((boxes[j][1] - boxes[j][0])[:2]))
        low, high = boxes[support][0][:2], boxes[support][1][:2]
    relevant = [
        (j, a[:2], b[:2])
        for j, (a, b) in boxes.items()
        if j != support and b[2] > bottom + 0.0001 and a[2] < bottom + height + 0.006
    ]
    ids = np.array([r[0] for r in relevant], dtype=int)
    lowers = np.array([r[1] for r in relevant]).reshape(-1, 2)
    uppers = np.array([r[2] for r in relevant]).reshape(-1, 2)

    def clear_lane(start, end):
        lower = np.minimum(start, end)[:2] - RADIUS
        upper = np.maximum(start, end)[:2] + RADIUS
        if np.any(lower < low) or np.any(upper > high):
            return False
        nearby = ids[((lowers < upper + 0.005) & (uppers > lower - 0.005)).all(axis=1)]
        if not len(nearby):
            return True
        # Half the sample spacing plus 1 mm clearance bounds between-sample distances.
        n = int(np.ceil(np.linalg.norm(end - start) / 0.006))
        margin = np.linalg.norm(end - start) / (2 * n) + 0.001
        for p in np.linspace(start, end, n + 1):
            data.qpos[:] = initial.qpos
            data.mocap_pos[mid] = p
            mujoco.mj_fwdPosition(model, data)
            if any(
                mujoco.mj_geomDistance(model, data, block, int(j), margin + 0.001, None) <= margin
                for j in nearby
            ):
                return False
        return True

    return clear_lane


def build(env, state, poses, xyz, fixed):
    """Use the fixed-case cylinder and prefer its existing point of intersection."""
    clear_lane = lane_checker(env, state, fixed)
    sim = env.sim
    sim.forward()
    model = sim.model._model
    block = sim.model.geom_name2id(NAME)
    mid = model.body_mocapid[sim.model.body_name2id(NAME)]
    height = fixed.get("height_m", HEIGHTS[0])
    set_height(model, block, height)
    robots = set(robot_geoms(env))
    data = mujoco.MjData(model)

    def update(q, pos):
        data.qpos[:] = q
        data.mocap_pos[mid] = pos
        mujoco.mj_fwdPosition(model, data)

    def robot_hit():
        for c in data.contact:
            a, b = int(c.geom1), int(c.geom2)
            other = b if a == block else a if b == block else -1
            if other in robots and c.dist < -0.001:
                return sim.model.geom_id2name(other), c.pos.tolist(), float(-c.dist)
        return None

    fixed_center = np.array(fixed["center"])
    k0 = fixed["reference_step"]

    def candidates():
        yield from (
            (k, fixed_center)
            for k in sorted(range(1, len(poses)), key=lambda k: abs(k - k0))
            if np.linalg.norm(xyz[k, :2] - fixed_center[:2]) < 0.14
        )
        for k in range(14, len(poses), 2):
            for dx, dy in [(0, 0), (0.025, 0), (-0.025, 0), (0, 0.025), (0, -0.025)]:
                yield k, np.r_[xyz[k, :2] + [dx, dy], fixed_center[2]]
        # A fingertip can travel beside the gripper center in confined scenes.
        for k in range(12, len(poses)):
            for radius in (0.02, 0.045, 0.065, 0.085):
                for angle in np.arange(16) * np.pi / 8:
                    xy = xyz[k, :2] + radius * np.array([np.cos(angle), np.sin(angle)])
                    yield k, np.r_[xy, fixed_center[2]]

    for k, center in candidates():
        update(poses[k], center)
        hit = robot_hit()
        if hit is None:
            continue
        heading = xyz[min(k + 2, len(xyz) - 1), :2] - xyz[max(k - 2, 0), :2]
        angle = np.arctan2(heading[1], heading[0]) + np.pi / 2
        for lead, tail in [
            (0.08, 0.08),
            (0.06, 0.06),
            (0.045, 0.055),
            (0.035, 0.045),
        ]:
            if k * DT - lead / SPEEDS[0] < 0.05:
                continue
            for deg in (0, 180, 30, 210, -30, 150, 55, 235, -55, 125):
                theta = angle + np.deg2rad(deg)
                direction = np.array([np.cos(theta), np.sin(theta), 0])
                start, end = center - direction * lead, center + direction * tail
                if not clear_lane(start, end):
                    continue
                # A stationary object at the entry point must miss the entire approach.
                for pose in poses[: k + 1]:
                    update(pose, start)
                    if any(
                        block in (c.geom1, c.geom2)
                        and c.dist <= 0
                        and (c.geom1 in robots or c.geom2 in robots)
                        for c in data.contact
                    ):
                        break
                else:
                    return dict(
                        key=fixed["key"],
                        start=start.tolist(),
                        end=end.tolist(),
                        center=center.tolist(),
                        height_m=height,
                        lead_m=lead,
                        step=k,
                        geom=hit[0],
                        contact=hit[1],
                        penetration_m=hit[2],
                    )
    raise ValueError("No supported timed crossing for this reference")
