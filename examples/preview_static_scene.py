"""Draw a task-order reference route on one official static SafeLIBERO episode."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import ImageDraw
from scipy.ndimage import distance_transform_edt
from libero.libero import benchmark
from libero.libero.envs.task_path import task_waypoints
from robosuite.utils.camera_utils import get_camera_transform_matrix
from preview_level_iii import project
from scene_regions import SceneRegions, make_preview_env


SIZE = 400
SETTLING_STEPS = 20


def frame_route(env, fovy, xyz):
    sim = env.sim
    camera = sim.model.camera_name2id('agentview')
    def pixels():
        matrix = get_camera_transform_matrix(sim, 'agentview', SIZE, SIZE)
        return np.array([project(matrix, p) for p in xyz])
    sim.model._model.cam_fovy[camera] = fovy
    scale = max(1., (abs(pixels() - SIZE/2) / [174, 154]).max())
    sim.model._model.cam_fovy[camera] = np.rad2deg(2*np.arctan(scale*np.tan(np.deg2rad(fovy)/2)))
    return pixels()


def set_view(env, cam, center, azimuth, elevation):
    """Point the world-fixed gallery camera at the task from an oblique angle."""
    import mujoco
    model = env.sim.model._model
    az, el = np.deg2rad([azimuth, elevation])
    back = np.array([np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0.])
    axes = np.column_stack([right, np.cross(back, right), back])
    if model.cam_bodyid[cam] != 0:
        raise ValueError('Expected a world-fixed gallery camera')
    model.cam_pos[cam] = center + .82*back
    mujoco.mju_mat2Quat(model.cam_quat[cam], axes.ravel())
    env.sim.forward()


def crossing(uv, mask):
    """Find a continuous interior chord of the visible obstacle on a route segment."""
    best = None
    depth = distance_transform_edt(mask)
    for j, (a, b) in enumerate(zip(uv, uv[1:])):
        length = np.linalg.norm(b-a)
        if length < 20:
            continue
        direction = (b-a)/length
        points = a + np.arange(0, length+0.1)[:, None]*direction
        x, y = np.rint(points).astype(int).T
        inside = mask[y, x]
        edges = np.diff(np.r_[0, inside, 0])
        for start, end in zip(np.flatnonzero(edges==1), np.flatnonzero(edges==-1)):
            if start < 4 or end+4 > len(points):
                continue
            # Reject scratches along the outline, even if they look long in projection.
            if np.count_nonzero(depth[y[start:end], x[start:end]] >= 6) < 12:
                continue
            if best is None or end-start > best[0]:
                best = (int(end-start), j, (points[start]+points[end-1])/2)
    return best


def draw_route(image, uv, hit):
    draw = ImageDraw.Draw(image)
    for j, (a, b) in enumerate(zip(uv, uv[1:])):
        length = np.linalg.norm(b-a)
        if length < 1:
            continue
        unit = (b-a)/length
        # Center a bright dash inside the obstacle silhouette.
        phase = (np.linalg.norm(hit[2]-a)-4) % 12 if hit and j == hit[1] else 0
        for d in np.arange(phase-12, length, 12):
            if d+8 <= 0:
                continue
            line = [tuple(a+max(0, d)*unit), tuple(a+min(d+8, length)*unit)]
            draw.line(line, fill='#483814', width=5)
            draw.line(line, fill='#ffdf3e', width=3)
    for j, (x, y) in enumerate(uv):
        if all(np.linalg.norm(uv[j]-p)>5 for p in uv[:j]):
            draw.ellipse((x-3,y-3,x+3,y+3), fill='#ffdf3e', outline='#483814')
            order = '/'.join(str(i) for i,p in enumerate(uv) if np.linalg.norm(uv[j]-p)<=5)
            draw.text((x+4,y-12), order, fill='#fff4ab', stroke_width=1, stroke_fill='#302c21')


def render(suite, level, task, out):
    # This historical preview selects from the original VLSA static scenes.
    original_level = "II-legacy" if level == "II" else level
    suite_obj = benchmark.get_benchmark_dict()['safelibero_'+suite](safety_level=original_level)
    env = make_preview_env(bddl_file_name=suite_obj.get_task_bddl_file_path(task),
                           safety_level=original_level, use_camera_obs=False, hard_reset=False)
    try:
        env.seed(13)
        env.reset()
        cam = env.sim.model.camera_name2id('agentview')
        fovy = env.sim.model._model.cam_fovy[cam]
        model = env.sim.model._model
        original_pose = model.cam_pos[cam].copy(), model.cam_quat[cam].copy()
        views = [None]
        if level == 'II':
            views.extend((az, el) for el in (50, 65, 35) for az in (-40, 40, 90, -90, 145, -145, 0, 180))
        best = None
        for i, state in enumerate(suite_obj.get_task_init_states(task)):
            env.set_init_state(state)
            for _ in range(SETTLING_STEPS):
                env.step([0.] * 6 + [-1.])
            labels, xyz = task_waypoints(env.env)
            for view in views:
                if view is None:
                    model.cam_pos[cam], model.cam_quat[cam] = original_pose
                    env.sim.forward()
                else:
                    center = .5*(xyz.min(axis=0) + xyz.max(axis=0))
                    set_view(env, cam, center, *view)
                uv = frame_route(env, fovy, xyz)
                regions = SceneRegions(env)
                image = regions.frame(SIZE)
                if min(regions.seen) < 20:
                    continue
                hit = crossing(uv, regions.obstacle_mask) if level == 'II' else None
                width = hit[0] if hit else 0
                if best is None or width > best[0]:
                    best = (width, i, labels, uv, regions.seen, image, hit, view)
                if level == 'I' or width >= 32:
                    break
            if best and (level == 'I' or best[0] >= 32):
                break
        if best is None:
            raise ValueError('No state shows both roles')
        width, i, labels, uv, visible, image, hit, view = best
        if level == 'II' and width < 10:
            raise ValueError(f'No clear projected crossing for {suite}-{task}; best {width} px')
        draw_route(image, uv, hit)
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out, quality=93)
        meta = dict(suite=suite, task=task, level=level, name=suite_obj.get_task(task).name,
                    state=i, settling_steps=SETTLING_STEPS, visible_pixels=visible.tolist(), waypoint_names=labels, waypoints_px=uv.tolist())
        if view:
            meta['view_azimuth_elevation_deg'] = list(view)
        if hit:
            meta.update(crossing_segment=hit[1], crossing_px=hit[2].tolist(), obstacle_chord_px=width)
        out.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', required=True)
    p.add_argument('--task', required=True, type=int)
    p.add_argument('--level', choices=['I','II'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    render(a.suite, a.level, a.task, a.output)
