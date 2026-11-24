"""Render the recorded approach with the frozen event on the same simulation clock."""

import numpy as np
from PIL import ImageDraw
from robosuite.utils.camera_utils import get_camera_transform_matrix

from libero.libero.envs.timed_crossing import DT, SPEEDS
from scene_regions import paired_preview


def render(env, state, case, poses, xyz, prefix):
    size = 352
    sim = env.sim
    cam = sim.model.camera_name2id("agentview")
    fovy = sim.model._model.cam_fovy[cam]
    for speed in SPEEDS:
        env.obstacle_speed_mps = speed
        env.set_init_state(state)
        event = env.env.level_three_motion
        sim.model._model.cam_fovy[cam] = fovy
        camera = get_camera_transform_matrix(sim, "agentview", size, size)
        corners = [
            point + [dx, dy, dz]
            for point in (event.start, event.end)
            for dx in (-0.024, 0.024)
            for dy in (-0.024, 0.024)
            for dz in (-case["height_m"] / 2, case["height_m"] / 2)
        ]
        scene = np.vstack((xyz, corners))
        uv = (camera @ np.c_[scene, np.ones(len(scene))].T).T
        scale = max(1.0, np.max(np.abs(uv[:, :2] / uv[:, 2, None] - size / 2)) / (0.43 * size))
        sim.model._model.cam_fovy[cam] = np.rad2deg(
            2 * np.arctan(scale * np.tan(np.deg2rad(fovy) / 2))
        )
        sim.forward()
        camera = get_camera_transform_matrix(sim, "agentview", size, size)

        def pixel(point):
            value = camera @ np.r_[point, 1.0]
            return tuple(value[:2] / value[2])

        first = max(0, int(np.floor((event.onset - 0.25) / DT)))
        last = min(len(poses) - 1, int(np.ceil((event.onset + event.length / speed + 0.25) / DT)))
        frames = []
        for i in range(first, last + 1):
            sim.data.time = event.policy_start + i * DT
            sim.data.qpos[:] = poses[i]
            event.apply(sim)
            at_witness = i == case["step"]
            image = paired_preview(env, xyz, case if at_witness else None, size=size)
            draw = ImageDraw.Draw(image)
            # Track the upper face of the cylinder from its parked start.
            top = [0, 0, case["height_m"] / 2]
            start = pixel(event.start + top)
            pos = pixel(event.position(i * DT) + top)
            draw.line([start, pos], fill="#173942", width=4)
            draw.line([start, pos], fill="#6af4ea", width=2)
            x, y = pos
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#70fff2", outline="#173942")
            if at_witness and speed == 0.10:
                still = image.copy()
                label = ImageDraw.Draw(still)
                label.rounded_rectangle((5, size - 23, 159, size - 5), radius=3, fill="#172630")
                label.text((9, size - 21), f"Reference contact   {i * DT:.2f} s", fill="white")
                still.save(str(prefix) + "-hit.webp", quality=88)
            text = f"{int(round(speed * 100))} cm/s   {i * DT:.2f} s"
            draw.rounded_rectangle((5, size - 23, 113, size - 5), radius=3, fill="#172630")
            draw.text((9, size - 21), text, fill="white")
            frames.append(image)
        frames[0].save(
            str(prefix) + f"-{round(speed * 100)}.webp",
            save_all=True,
            append_images=frames[1:],
            duration=50,
            loop=0,
            quality=74,
            method=3,
        )
