"""Render a Level III episode with a hold command, without loading a policy."""
import argparse
import json
from pathlib import Path

import numpy as np
from scene_regions import SceneRegions, make_preview_env
from robosuite.utils.camera_utils import get_camera_transform_matrix
from PIL import Image, ImageDraw
from libero.libero.envs.task_path import segment_hit
from libero.libero import benchmark


def project(matrix, xyz):
    p = matrix @ np.r_[xyz, 1.]
    return p[:2] / p[2]


def dashed(draw, a, b):
    length = np.linalg.norm(b - a)
    if length < 1:
        return
    unit = (b-a) / length
    for d in np.arange(0., length, 10.):
        line = [tuple(a+d*unit), tuple(a+min(d+6., length)*unit)]
        draw.line(line, fill='#4b3c20', width=4)
        draw.line(line, fill='#ffdb45', width=2)


def render(suite_name, task_id, index, speed, delay, output, crossing=False):
    suite = benchmark.get_benchmark_dict()['safelibero_' + suite_name](safety_level='III')
    env = make_preview_env(bddl_file_name=suite.get_task_bddl_file_path(task_id),
                             safety_level='III', obstacle_speed_mps=speed, cross_task_path=crossing,
                             obstacle_start_s=delay, use_camera_obs=False,
                             camera_names=['agentview'], camera_heights=320, camera_widths=320,
                             control_freq=20, hard_reset=False)
    try:
        env.seed(13)
        env.reset()
        states = suite.get_task_init_states(task_id)
        projection = get_camera_transform_matrix(env.sim, 'agentview', 320, 320)
        indices = range(len(states)) if index is None else [index]
        for index in indices:
            try:
                env.set_init_state(states[index])
            except ValueError as exc:
                if str(exc).startswith('No clear 16 cm obstacle lane'):
                    continue
                raise
            motion = env.env.level_three_motion
            if crossing:
                uv = [project(projection, p) for p in motion.path]
                a, b = uv[motion.segment:motion.segment+2]
                lane = project(projection, motion.start+motion.delta) - project(projection, motion.start)
                if abs(np.cross(b-a, lane)) / max(1., np.linalg.norm(b-a)) < 17:
                    continue
            break
        else:
            raise ValueError('No clearly visible crossing for the requested states')
        motion = env.env.level_three_motion
        body = env.sim.model.body_name2id(env.env.objects_dict[motion.name].root_body)
        projection = get_camera_transform_matrix(env.sim, 'agentview', 320, 320)
        marker_offset = np.zeros(3)
        if crossing:
            a, b = motion.path[motion.segment:motion.segment+2]
            for f in (.5, .375, .625, .25, .75):
                shift = f * motion.delta
                hit = segment_hit(env.sim.model._model, env.sim.data._data, motion.geoms, a, b, shift)
                if hit is not None:
                    marker_offset = hit - motion.start - shift
                    break
            else:
                raise RuntimeError('Verified crossing marker missing')
            ends = [motion.start+marker_offset, motion.start+marker_offset+motion.delta]
            screen = np.array([project(projection, p) for p in [*motion.path, *ends]])
            scale = max(1., np.max(abs(screen-160)/[140., 123.]))
            cam = env.sim.model.camera_name2id('agentview')
            fovy = env.sim.model._model.cam_fovy[cam]
            env.sim.model._model.cam_fovy[cam] = np.rad2deg(2*np.arctan(scale*np.tan(np.deg2rad(fovy)/2)))
            projection = get_camera_transform_matrix(env.sim, 'agentview', 320, 320)
            uv = [project(projection, p) for p in motion.path]
            lane_uv = [tuple(project(projection, p)) for p in ends]
        regions = SceneRegions(env)
        frames, errors, hits = [], [], []
        # A static control uses the same viewing duration as the default motion.
        duration = delay + motion.length / (speed or .10) + .5
        stride = 1 if speed >= .40 else 2
        for step in range(int(np.ceil(duration * 20)) + 1):
            elapsed = env.sim.data.time - motion.t0
            travel = min(motion.length, max(0., elapsed - delay) * speed)
            expected = motion.start + motion.delta * travel / motion.length
            errors.append(np.linalg.norm(env.sim.data.body_xpos[body] - expected))
            if step % stride == 0:
                frame = regions.frame(320)
                draw = ImageDraw.Draw(frame)
                if crossing:
                    draw.line(lane_uv, fill='#22494a', width=3)
                    draw.line(lane_uv, fill='#53e5df', width=1)
                    for a, b in zip(uv, uv[1:]):
                        dashed(draw, a, b)
                    for k, (x, y) in enumerate(uv):
                        draw.ellipse((x-3,y-3,x+3,y+3), fill='#ffdb45', outline='#463b20')
                        # Shared destinations keep one marker for both visits.
                        if all(np.linalg.norm(np.array([x,y])-v) > 8 for v in uv[:k]):
                            numbers = '/'.join(str(n) for n, v in enumerate(uv) if np.linalg.norm(uv[k]-v) < 1.)
                            draw.text((x+4,y-12), numbers, fill='#fff4ab',
                                      stroke_width=1, stroke_fill='#302c21')
                    a, b = motion.path[motion.segment:motion.segment+2]
                    hit = segment_hit(env.sim.model._model, env.sim.data._data,
                                      motion.geoms, a, b, np.zeros(3))
                    if hit is not None:
                        hits.append(len(frames))
                        hx, hy = project(projection, hit)
                        draw.ellipse((hx-7, hy-7, hx+7, hy+7), outline='#ffec79', width=3)
                point = projection @ np.r_[env.sim.data.body_xpos[body] + marker_offset, 1.]
                x, y = (point[:2] / point[2]).round().astype(int)
                draw.ellipse((x-5, y-5, x+5, y+5), outline=(47, 236, 226), width=2)
                draw.rectangle((0, 293, 320, 320), fill=(27, 34, 38))
                draw.text((10, 301), f'{motion.name.split("_obstacle")[0].replace("_", " ")}  |  {speed*100:g} cm/s  |  {elapsed:.1f} s', fill='white')
                if crossing and hit is not None:
                    draw.text((252,301), 'CROSSING', fill='#ffdf4e')
                frames.append(frame)
            if step < int(np.ceil(duration * 20)):
                env.step([0.] * 6 + [-1.])
        output.parent.mkdir(exist_ok=True, parents=True)
        # A shared palette keeps stationary background pixels identical across frames.
        sample = Image.new('RGB', (80, 80 * len(frames)))
        for frame_number, frame in enumerate(frames):
            sample.paste(frame.resize((80, 80)), (0, 80 * frame_number))
        palette = sample.quantize(colors=256)
        indexed = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
        indexed[0].save(output, save_all=True, append_images=indexed[1:],
                        duration=50 * stride, loop=0, optimize=False)
        meta = dict(suite=suite_name, task=task_id, state=index, hazard=motion.name,
                    speed_mps=speed, frame_ms=50*stride, delay_s=delay, start_xyz=motion.start.tolist(),
                    end_xyz=(motion.start + motion.delta).tolist(),
                    max_position_error_m=float(max(errors)), frames=len(frames))
        assert min(regions.seen) > 0, 'Missing target or obstacle pixels'
        meta['visible_pixels_max'] = regions.seen.tolist()
        frames[0].save(output.with_name(output.stem+'-motion.webp'),save_all=True,
                       append_images=frames[1:],duration=50*stride,loop=0,quality=85,method=2)
        if crossing:
            if speed and (not hits or hits[0] == 0 or hits[-1] == len(frames)-1):
                raise ValueError('Expected clear, crossing, clear in the rendered frames')
            still = hits[len(hits)//2] if hits else 0
            frames[still].save(output.with_suffix('.webp'), lossless=True)
            meta.update(waypoint_names=motion.labels, waypoints_xyz=motion.path.tolist(),
                        waypoints_px=np.asarray(uv).tolist(), obstacle_line_px=lane_uv,
                        crossing_segment=motion.segment, crossing_frames=hits, still_frame=still)
        output.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', required=True, choices=['spatial', 'object', 'goal', 'long'])
    p.add_argument('--task', type=int, choices=range(4), required=True)
    p.add_argument('--state', type=int, help='fixed initial state (default: first eligible state)')
    p.add_argument('--cross-task-path', action='store_true', help='select a lane crossing the task reference path')
    p.add_argument('--speed', type=float, default=.10, help='metres per second')
    p.add_argument('--delay', type=float, default=.60, help='seconds after initialization')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    render(a.suite, a.task, a.state, a.speed, a.delay, a.output, a.cross_task_path)
