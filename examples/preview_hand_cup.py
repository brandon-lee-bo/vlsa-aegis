"""Render and check the Level III hand placing a mug beside the plate."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from robosuite.utils.camera_utils import get_camera_transform_matrix
from libero.libero import benchmark
from libero.libero.envs.task_path import segment_hit
from scene_regions import SceneRegions, make_preview_env
from preview_level_iii import project, dashed


def render(speed, output, suite_name, task, state):
    suite = benchmark.get_benchmark_dict()['safelibero_'+suite_name](safety_level='III-legacy')
    env = make_preview_env(bddl_file_name=suite.get_task_bddl_file_path(task), safety_level='III-legacy',
                             hand_places_cup=True, obstacle_speed_mps=speed,
                             obstacle_start_s=.4, use_camera_obs=False, hard_reset=False)
    try:
        env.seed(13)
        env.reset()
        env.set_init_state(suite.get_task_init_states(task)[state])
        sim=env.sim
        m,d=sim.model._model,sim.data._data
        motion=env.env.level_three_motion
        proj=get_camera_transform_matrix(sim,'agentview',400,400)
        uv=[project(proj,pt) for pt in motion.path]
        regions = SceneRegions(env)
        frames,hits,settled=[],[],[]
        drift={}
        bodies={n:b for n,b in env.env.obj_body_id.items() if n in env.env.objects_dict and 'obstacle' not in n and n != 'box_base_1'}
        excursion={n:0. for n in bodies}
        init={n:d.xpos[b].copy() for n,b in bodies.items()}
        stages={}
        for step in range(int(np.ceil(motion.duration*20))+1):
            t=d.time-motion.t0
            for n,b in bodies.items():
                excursion[n]=max(excursion[n],float(np.linalg.norm(d.xpos[b]-init[n])))
            if step%2==0:
                img=regions.frame(400)
                draw=ImageDraw.Draw(img)
                for a,b in zip(uv,uv[1:]):
                    dashed(draw,a,b)
                for i,(x,y) in enumerate(uv):
                    draw.ellipse((x-3,y-3,x+3,y+3),fill='#ffdb45',outline='#463b20')
                    draw.text((x+5,y-13),str(i),fill='#ffe878',stroke_width=1,stroke_fill='#383124')
                for a,b in zip(motion.path,motion.path[1:]):
                    hit=segment_hit(m,d,motion.hand_geoms,a,b,np.zeros(3))
                    if hit is not None:
                        x,y=project(proj,hit)
                        draw.ellipse((x-7,y-7,x+7,y+7),outline='#ffde53',width=3)
                        hits.append(len(frames))
                        break
                draw.rectangle((0,371,400,400),fill='#192c36')
                draw.text((12,380),motion.phase,fill='white')
                draw.text((325,380),f'{t:.1f} s',fill='#d8e8e8')
                stages.setdefault(motion.phase,len(frames))
                frames.append(img)
            if t>motion.release_s+.35:
                settled.append(d.xpos[motion.cup].copy())
            env.step([0.]*6+[-1.])
        for n,b in bodies.items():
            drift[n]=float(np.linalg.norm(d.xpos[b]-init[n]))
        final=d.xpos[motion.cup].copy()
        assert min(regions.seen)>0
        report=dict(visible_pixels_max=regions.seen.tolist(),state=state,task=f'{suite_name}-{task}',speed_mps=speed,release_s=motion.release_s,phase_frames=stages,
                    hand_crossing_frames=hits,frames=len(frames),cup_drop_xyz=motion.drop.tolist(),
                    cup_final_xyz=final.tolist(),cup_drop_error_m=float(np.linalg.norm(final-motion.drop)),
                    settled_drift_m=float(np.linalg.norm(np.array(settled)-final,axis=1).max()),
                    other_object_net_shift_m=drift,other_object_peak_shift_m=excursion,robot_hand_contact=bool(motion.touched_hand),
                    robot_cup_contact=bool(motion.touched_cup))
        output.parent.mkdir(parents=True,exist_ok=True)
        output.with_suffix('.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        sample=Image.new('RGB',(96,96*len(frames)))
        for i,im in enumerate(frames):
            sample.paste(im.resize((96,96)),(0,96*i))
        pal=sample.quantize(colors=256)
        imgs=[im.quantize(palette=pal,dither=Image.Dither.NONE) for im in frames]
        imgs[0].save(output,save_all=True,append_images=imgs[1:],duration=100,loop=0,optimize=True)
        frames[0].save(output.with_name(output.stem+'-motion.webp'),save_all=True,
                       append_images=frames[1:],duration=100,loop=0,quality=87,method=3)
        if hits:
            frames[hits[len(hits)//2]].save(output.with_suffix('.webp'),lossless=True)
        assert hits and hits[0]>0 and hits[-1]<len(frames)-1
        assert 'Cup remains' in stages and report['cup_drop_error_m']<.012 and report['settled_drift_m']<.004
        assert max(excursion.values())<.005 and not motion.touched_hand and not motion.touched_cup
    finally:
        env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['spatial','goal'],default='spatial')
    p.add_argument('--task',type=int,default=0)
    p.add_argument('--state',type=int,default=4)
    p.add_argument('--speed',type=float,default=.20)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    render(args.speed,args.output,args.suite,args.task,args.state)
