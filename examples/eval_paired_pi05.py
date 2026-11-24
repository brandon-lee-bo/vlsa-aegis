"""Run paired SafeLIBERO-Pro episodes using a local OpenPI policy."""
import argparse
import dataclasses
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from libero.libero import benchmark
from libero.libero.envs.paired_obstacle import build_case, start_contact_monitor
from openpi.policies import policy_config
from openpi.training import config
from openpi_client import image_tools
from robosuite.utils.transform_utils import quat2axisangle
from scene_regions import make_preview_env, paired_preview

LIMITS = {"spatial":220, "object":280, "goal":300, "long":520}


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rollout(env, state, prompt, policy, limit, seed, pause_steps=0):
    env.seed(seed)
    obs = env.set_init_state(state)
    for _ in range(10):
        obs,_,_,_ = env.step([0.]*6+[-1.])
    settled = env.sim.get_state().flatten().copy()
    reset_seed(seed)
    hit = start_contact_monitor(env)
    queue = deque()
    qpos,xyz = [env.sim.data.qpos.copy()], [obs["robot0_eef_pos"].copy()]
    for step in range(limit):
        if step < pause_steps:
            queue.append(np.array([0., 0., 0., 0., 0., 0., -1.]))
        elif not queue:
            def image(key):
                return image_tools.convert_to_uint8(image_tools.resize_with_pad(
                    np.ascontiguousarray(obs[key][::-1,::-1]),224,224))
            request = {"observation/image":image("agentview_image"),
                       "observation/wrist_image":image("robot0_eye_in_hand_image"),
                       "observation/state":np.r_[obs["robot0_eef_pos"],
                          quat2axisangle(obs["robot0_eef_quat"].copy()),obs["robot0_gripper_qpos"]],
                       "prompt":prompt}
            actions = policy.infer(request)["actions"][:5]
            assert actions.shape == (5,7) and np.isfinite(actions).all()
            queue.extend(actions)
        obs,_,done,_ = env.step(queue.popleft().tolist())
        qpos.append(env.sim.data.qpos.copy())
        xyz.append(obs["robot0_eef_pos"].copy())
        if done:
            break
    success = bool(env.check_success())
    return dict(success=success, safe_success=success and not any(hit.values()),
                contacts=hit, steps=len(qpos)-1), np.asarray(qpos), np.asarray(xyz), settled


def evaluate(args):
    suite = benchmark.get_benchmark_dict()["safelibero_"+args.suite](safety_level="I")
    task = suite.get_task(args.task)
    states = suite.get_task_init_states(args.task)
    cfg = config.get_config("pi05_libero")
    cfg = dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,pytorch_compile_mode=None))
    policy = policy_config.create_trained_policy(cfg,args.checkpoint,pytorch_device="cuda")
    args.output.mkdir(parents=True,exist_ok=True)
    for episode in args.episodes:
        stem = f"{args.suite}-{args.task}-{episode}"
        path = args.output/(stem+".json")
        if path.exists():
            continue
        state = states[episode]
        seed = 7 + episode
        results = {}
        for level in ("I","II"):
            reset_seed(seed)
            env = make_preview_env(bddl_file_name=suite.get_task_bddl_file_path(args.task),
                camera_names=["agentview","robot0_eye_in_hand"],camera_heights=256,camera_widths=256,
                hard_reset=False,control_freq=20,safety_level=level)
            try:
                env.reset()
                if level == "II":
                    np.savez_compressed(args.output/(stem+".npz"),level_i_qpos=poses,level_i_xyz=xyz)
                    case = build_case(env,state,poses,xyz)
                    env.paired_case = case
                metrics,q,p,settled = rollout(env,state,task.language,policy,LIMITS[args.suite],seed)
                results[level] = metrics
                if level == "I":
                    poses,xyz,initial = q,p,settled
                else:
                    np.testing.assert_allclose(initial,settled,atol=1e-8,rtol=0)
                if episode == 0:
                    env.regenerate_obs_from_state(settled)
                    paired_preview(env,xyz,case if level=="II" else None,args.output/f"{level}-{stem}.webp")
            finally:
                env.close()
        row = dict(suite=args.suite,task=args.task,name=task.name,episode=episode,seed=seed,
                   case=case,results=results)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(row,indent=2))
        temp.replace(path)
        print("PAIR",stem,results,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",choices=LIMITS,required=True)
    parser.add_argument("--task",type=int,choices=range(4),required=True)
    parser.add_argument("--episodes",type=int,nargs="+",default=[0,1,2])
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--output",type=Path,required=True)
    with torch.inference_mode():
        torch.set_num_threads(1)
        evaluate(parser.parse_args())
