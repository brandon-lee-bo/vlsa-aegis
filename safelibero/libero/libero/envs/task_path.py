"""Task-order waypoints and transverse lanes for the Level III gallery."""
from pathlib import Path

import mujoco
import numpy as np


def task_waypoints(domain):
    sim = domain.sim
    m, d = sim.model._model, sim.data._data
    points = [d.site_xpos[domain.robots[0].eef_site_id].copy()]
    labels = ['Start']
    for _, obj, goal in domain.parsed_problem['goal_state']:
        pick = domain.object_states_dict[obj].get_geom_state()['pos'].copy()
        place = domain.object_states_dict[goal].get_geom_state()['pos'].copy()
        if goal == 'wooden_cabinet_1_top_region':
            joint = sim.model.joint_name2id('wooden_cabinet_1_top_level')
            handle = next(i for i in range(m.ngeom) if m.geom_dataid[i] >= 0
                          and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                          and 'top_handle' in m.mesh(m.geom_dataid[i]).name)
            center, _ = m.geom_aabb[handle].reshape(2, 3)
            grip = d.geom_xpos[handle] + d.geom_xmat[handle].reshape(3, 3) @ center
            pull = d.xaxis[joint] * (m.jnt_range[joint, 0] - d.qpos[m.jnt_qposadr[joint]])
            points.extend([grip, grip + pull])
            labels.extend(['Handle', 'Open'])
            place += pull
        points.append(pick)
        labels.append(obj)
        # For short transfers on the cluttered tabletop, lift the bowl on
        # its way out before approaching the receiving surface from the side.
        if Path(domain.bddl_file_name).stem in ('put_the_bowl_on_the_plate', 'put_the_bowl_on_top_of_the_cabinet'):
            v = place[:2] - pick[:2]
            side = np.array([-v[1], v[0]]) / max(np.linalg.norm(v), 1e-8)
            mid = (pick+place)/2
            mid[:2] += (.20 if goal == 'plate_1' else .24) * side
            mid[2] = pick[2] + .04
            points.append(mid)
            labels.append('Carry')
        points.append(place)
        labels.append(goal)
    return labels, np.asarray(points)


def segment_hit(model, data, geoms, a, b, shift):
    length = np.linalg.norm(b - a)
    if length < 1e-8:
        return None
    direction = (b - a) / length
    best = length
    for g in geoms:
        kind = model.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            distance = mujoco.mj_rayMesh(model, data, g, a - shift, direction)
        else:
            distance = mujoco.mju_rayGeom(data.geom_xpos[g], data.geom_xmat[g],
                model.geom_size[g], a - shift, direction, kind)
        if 0 <= distance < best:
            best = distance
    if best < length:
        return a + direction * best
    return None


def crossing_lane(starts, vector, z, origin, lower, upper, points, model, data, geoms):
    """Rank supported lanes that enter AND leave a finite 3-D task segment."""
    best = None
    axis = int(np.argmax(abs(vector)))
    other = 1 - axis
    for j, (a, b) in enumerate(zip(points, points[1:])):
        v = b - a
        if abs(v[other]) < .035:
            continue
        f = (starts[:, other] - a[other]) / v[other]
        cross = a + f[:, None] * v
        t = (cross[:, axis] - starts[:, axis]) / vector[axis]
        good = ((f > .10) & (f < .90) & (t > .23) & (t < .77)
                & (cross[:, 2] > z + lower[2]) & (cross[:, 2] < z + upper[2]))
        candidates = np.flatnonzero(good)
        cost = (abs(f - .5) * .3 + abs(t - .5) +
                abs(cross[:, 2] - z - .5*(upper[2]+lower[2])) * 2 +
                .15 * np.linalg.norm(starts-origin[:2], axis=1))
        delta = np.r_[vector, 0.]
        for i in candidates[np.argsort(cost[candidates])]:
            if best is not None and cost[i] >= best[0]:
                break
            start = np.r_[starts[i], z]
            at = lambda fraction: segment_hit(model, data, geoms, a, b, start + fraction*delta - origin)
            if at(0) is None and at(1) is None and at(round(t[i]*8)/8) is not None:
                best = (cost[i], start, delta, j)
                break
    return best
