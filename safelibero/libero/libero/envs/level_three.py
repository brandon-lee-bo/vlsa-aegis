"""Deterministic, table-supported linear obstacle motion for SafeLIBERO Level III."""

import numpy as np
import mujoco


def _bounds(model, data, geom):
    # MuJoCo's local collision bounding box also covers translated mesh origins.
    center, half = model.geom_aabb[geom].reshape(2, 3)
    rotation = data.geom_xmat[geom].reshape(3, 3)
    mid = data.geom_xpos[geom] + rotation @ center
    extent = abs(rotation) @ half
    return mid - extent, mid + extent


def _route(domain, name, path=None):
    sim = domain.sim
    m, d = sim.model._model, sim.data._data
    body = sim.model.body_name2id(domain.objects_dict[name].root_body)
    origin = d.xpos[body].copy()
    boxes, own = {}, []
    for i in range(m.ngeom):
        if not (m.geom_contype[i] or m.geom_conaffinity[i]):
            continue
        boxes[i] = _bounds(m, d, i)
        parent = m.geom_bodyid[i]
        while parent and parent != body:
            parent = m.body_parentid[parent]
        if parent == body:
            own.append(i)
    lower = np.min([boxes[i][0] for i in own], axis=0) - origin
    upper = np.max([boxes[i][1] for i in own], axis=0) - origin
    height = domain.workspace_offset[2]
    if domain.workspace_name == 'floor':
        support = sim.model.geom_name2id('floor')
        lo, hi = np.array([-.46, -.46, 0.]), np.array([.46, .46, d.geom_xpos[support, 2]])
    else:
        surfaces = [i for i, (a, b) in boxes.items()
                    if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
                    and np.all(b[:2] - a[:2] >= .50) and abs(b[2] - height) < .09]
        if not surfaces:
            raise ValueError('No tabletop collision surface in this scene')
        support = max(surfaces, key=lambda i: boxes[i][1][2])
        lo, hi = boxes[support]
    z = hi[2] - lower[2] + .0005
    # Keep the whole swept box on the support and outside initial scene geometry.
    # The grid search gives the same lane for a given reset at every speed.
    low = np.maximum(lo[:2], -.43) - lower[:2]
    high = np.minimum(hi[:2], .43) - upper[:2]
    grid = np.arange(-.43, .431, .0125)
    starts = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
    blockers = [(a, b) for i, (a, b) in boxes.items() if i != support and i not in own
                and b[2] > z + lower[2] and a[2] < z + upper[2]]
    best = None
    for dx, dy in ((.16, 0), (-.16, 0), (0, .16), (0, -.16)):
        vector = np.array([dx, dy])
        first = starts + np.minimum(0, vector)
        last = starts + np.maximum(0, vector)
        valid = ((first >= low) & (last <= high)).all(axis=1)
        for a, b in blockers:
            overlap = ((first + lower[:2] < b[:2] + .002) &
                       (last + upper[:2] > a[:2] - .002)).all(axis=1)
            valid &= ~overlap
        if not valid.any():
            continue
        candidates = starts[valid]
        if path is not None:
            from .task_path import crossing_lane
            lane = crossing_lane(candidates, vector, z, origin, lower, upper, path, m, d, own)
            if lane is not None and (best is None or lane[0] < best[0]):
                best = lane
            continue
        cost = np.linalg.norm(candidates - origin[:2], axis=1)
        cost += .35 * np.linalg.norm(candidates + vector - np.array([-.08, 0.]), axis=1)
        i = cost.argmin()
        if best is None or cost[i] < best[0]:
            best = cost[i], np.r_[candidates[i], z], np.r_[vector, 0.]
    if best is None:
        raise ValueError('No clear 16 cm obstacle lane for this initial state')
    return best[1], best[2], body, own, best[3] if path is not None else None


class LinearObstacle:
    def __init__(self, domain, speed, delay, cross_task_path=False):
        if not np.isfinite([speed, delay]).all() or min(speed, delay) < 0:
            raise ValueError('Speed and start delay must be finite and non-negative')
        sim = domain.sim
        candidates = []
        for name, obj in domain.objects_dict.items():
            if 'obstacle' not in name:
                continue
            xyz = sim.data.body_xpos[sim.model.body_name2id(obj.root_body)]
            if np.max(np.abs(xyz[:2])) <= .53 and xyz[2] >= -.05:
                candidates.append(name)
        if not candidates:
            raise ValueError('No active obstacle in this Level I state')
        self.name = sorted(candidates)[0]
        self.labels, self.path = [], None
        if cross_task_path:
            from .task_path import task_waypoints
            self.labels, self.path = task_waypoints(domain)
        self.start, self.delta, body, self.geoms, self.segment = _route(domain, self.name, self.path)
        m, d = sim.model._model, sim.data._data
        joint = m.body_jntadr[body]
        if joint < 0 or m.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError('Level III needs a free obstacle body')
        self.qa = m.jnt_qposadr[joint]
        self.va = m.jnt_dofadr[joint]
        self.offset = d.qpos[self.qa:self.qa + 3].copy() - d.xpos[body]
        self.quat = d.qpos[self.qa + 3:self.qa + 7].copy()
        self.speed, self.delay = speed, delay
        self.t0 = float(d.time)
        self.length = np.linalg.norm(self.delta)

    def apply(self, sim):
        d = sim.data._data
        elapsed = max(0., float(d.time) - self.t0 - self.delay)
        distance = min(self.length, elapsed * self.speed)
        direction = self.delta / self.length
        d.qpos[self.qa:self.qa + 3] = self.start + self.offset + direction * distance
        d.qpos[self.qa + 3:self.qa + 7] = self.quat
        d.qvel[self.va:self.va + 6] = 0.
        if d.time >= self.t0 + self.delay and distance < self.length:
            d.qvel[self.va:self.va + 3] = direction * self.speed
        sim.forward()
