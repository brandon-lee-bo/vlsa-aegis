"""Annotate visible target and obstacle pixels using MuJoCo geometry IDs."""
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def make_preview_env(**kwargs):
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.errors import RandomizationError
    # Construction samples a disposable scene before the recorded state is restored.
    for attempt in range(5):
        try:
            return OffScreenRenderEnv(**kwargs)
        except RandomizationError:
            if attempt == 4:
                raise


class SceneRegions:
    def __init__(self, env):
        domain = env.env
        self.sim = env.sim
        model = self.sim.model._model
        goals = domain.parsed_problem['goal_state']
        targets = {domain.obj_body_id[obj] for _, obj, _ in goals}
        if any(dest == 'wooden_cabinet_1_top_region' for _, _, dest in goals):
            joint = self.sim.model.joint_name2id('wooden_cabinet_1_top_level')
            targets.add(model.jnt_bodyid[joint])
        hazards = {b for name,b in domain.obj_body_id.items() if 'obstacle' in name}
        self.roles = np.zeros(model.ngeom, np.uint8)
        for g in range(model.ngeom):
            b = model.geom_bodyid[g]
            if (self.sim.model.geom_id2name(g) or '').startswith('human_'):
                self.roles[g] = 2
            while b:
                if b in targets or b in hazards:
                    self.roles[g] = 1 if b in targets else 2
                    break
                b = model.body_parentid[b]
        self.seen = np.zeros(2, dtype=int)

    def frame(self, size):
        opts = dict(width=size, height=size, camera_name='agentview')
        rgb = self.sim.render(**opts)[::-1].copy()
        seg = self.sim.render(segmentation=True, **opts)[::-1]
        valid = (seg[:,:,0] == mujoco.mjtObj.mjOBJ_GEOM) & (seg[:,:,1] >= 0)
        roles = np.zeros((size,size), np.uint8)
        roles[valid] = self.roles[seg[:,:,1][valid]]
        self.obstacle_mask = roles == 2
        image = Image.fromarray(rgb)
        colors = [(76,190,255), (255,132,63)]
        for role,color in enumerate(colors, 1):
            mask = roles == role
            self.seen[role-1] = max(self.seen[role-1], int(mask.sum()))
            if not mask.any():
                continue
            area = Image.fromarray(mask.astype(np.uint8)*255)
            edge = np.asarray(area.filter(ImageFilter.MaxFilter(3))) - np.asarray(area.filter(ImageFilter.MinFilter(3)))
            overlay = Image.new('RGB', image.size, color)
            image.paste(overlay, mask=area.point(lambda v:v//6))
            image.paste(overlay, mask=Image.fromarray(edge))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((5,5,152,24),radius=4,fill='#172630')
        for x,c,label in [(12,colors[0],'Target'),(76,colors[1],'Obstacle')]:
            draw.rectangle((x,12,x+6,18),fill=c)
            draw.text((x+11,9),label,fill='white')
        return image
