"""Render a Level III episode with a hold command, without loading a policy."""
import argparse
import json
from pathlib import Path

import numpy as np
from robosuite.utils.camera_utils import get_camera_transform_matrix
from PIL import Image, ImageDraw
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


def render(suite_name, task_id, index, speed, delay, output):
    suite = benchmark.get_benchmark_dict()['safelibero_' + suite_name](safety_level='III')
    env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(task_id),
                             safety_level='III', obstacle_speed_mps=speed,
                             obstacle_start_s=delay, use_camera_obs=False,
                             camera_names=['agentview'], camera_heights=320, camera_widths=320,
                             control_freq=20, hard_reset=False)
    try:
        env.seed(13)
        env.reset()
        states = suite.get_task_init_states(task_id)
        env.set_init_state(states[index])
        motion = env.env.level_three_motion
        body = env.sim.model.body_name2id(env.env.objects_dict[motion.name].root_body)
        projection = get_camera_transform_matrix(env.sim, 'agentview', 320, 320)
        frames, errors = [], []
        # A static control uses the same viewing duration as the default motion.
        duration = delay + motion.length / (speed or .10) + .5
        for step in range(int(np.ceil(duration * 20)) + 1):
            elapsed = env.sim.data.time - motion.t0
            travel = min(motion.length, max(0., elapsed - delay) * speed)
            expected = motion.start + motion.delta * travel / motion.length
            errors.append(np.linalg.norm(env.sim.data.body_xpos[body] - expected))
            if step % 2 == 0:
                pixels = env.sim.render(width=320, height=320, camera_name='agentview')[::-1]
                frame = Image.fromarray(pixels.copy())
                draw = ImageDraw.Draw(frame)
                point = projection @ np.r_[env.sim.data.body_xpos[body], 1.]
                x, y = (point[:2] / point[2]).round().astype(int)
                draw.ellipse((x-5, y-5, x+5, y+5), outline=(255, 201, 78), width=2)
                draw.rectangle((0, 293, 320, 320), fill=(27, 34, 38))
                draw.text((10, 301), f'{motion.name.split("_obstacle")[0].replace("_", " ")}  |  {speed*100:g} cm/s  |  {elapsed:.1f} s', fill='white')
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
                        duration=100, loop=0, optimize=False)
        meta = dict(suite=suite_name, task=task_id, state=index, hazard=motion.name,
                    speed_mps=speed, delay_s=delay, start_xyz=motion.start.tolist(),
                    end_xyz=(motion.start + motion.delta).tolist(),
                    max_position_error_m=float(max(errors)), frames=len(frames))
        output.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', required=True, choices=['spatial', 'object', 'goal', 'long'])
    p.add_argument('--task', type=int, choices=range(4), required=True)
    p.add_argument('--state', type=int, default=0)
    p.add_argument('--speed', type=float, default=.10, help='metres per second')
    p.add_argument('--delay', type=float, default=.60, help='seconds after initialization')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    render(a.suite, a.task, a.state, a.speed, a.delay, a.output)
