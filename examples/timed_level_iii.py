"""Generate, verify, render or evaluate a time-matched SafeLIBERO-Pro crossing."""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from libero.libero import benchmark
from libero.libero.envs.paired_obstacle import NAME, frozen_cases, robot_geoms, state_key
from libero.libero.envs.timed_crossing import DT, SPEEDS, build, lane_checker
from scene_regions import make_preview_env

DOCS = Path(__file__).resolve().parents[1] / "docs"


def verify(env, state, case, poses):
    robots = set(robot_geoms(env))
    block = env.sim.model.geom_name2id(NAME)
    first_hits = {}
    earlier_object_contacts = {}
    model = env.sim.model._model
    roots = {
        env.env.obj_body_id[name]: name
        for name in env.env.objects_dict
        if name in env.env.obj_body_id
    }
    movable = {}
    for g, body in enumerate(model.geom_bodyid):
        while body and body not in roots:
            body = model.body_parentid[body]
        if body in roots:
            movable[g] = roots[body]
    for speed in SPEEDS:
        env.obstacle_speed_mps = speed
        env.set_init_state(state)
        sim = env.sim
        model, d = sim.model._model, sim.data._data
        motion = env.env.level_three_motion
        k = case["step"]
        np.testing.assert_allclose(d.geom_xpos[block], case["start"], atol=1e-9)
        assert not any(block in (c.geom1, c.geom2) and c.dist < -1e-5 for c in d.contact)
        recorded = None
        touched = set()
        for i in range(k + 1):
            d.time = motion.policy_start + i * DT
            d.qpos[:] = poses[i]
            motion.apply(sim)
            hits = [
                c
                for c in d.contact
                if block in (c.geom1, c.geom2)
                and (c.geom1 in robots or c.geom2 in robots)
                and c.dist <= 0
            ]
            if hits and recorded is None:
                recorded = i * DT
            if recorded is None:
                for c in d.contact:
                    if c.dist < 0 and block in (c.geom1, c.geom2):
                        other = c.geom2 if c.geom1 == block else c.geom1
                        if other in movable:
                            touched.add(movable[other])
            if i == k:
                np.testing.assert_allclose(d.geom_xpos[block], case["center"], atol=1e-9)
                assert any(-c.dist >= 0.001 for c in hits)
        assert motion.onset < recorded <= k * DT < motion.onset + motion.length / speed
        first_hits[str(speed)] = round(recorded, 4)
        earlier_object_contacts[str(speed)] = sorted(touched)
    # A time-shifted negative control: park at the entry point through the witness.
    env.obstacle_speed_mps = 0.0
    env.set_init_state(state)
    event = env.env.level_three_motion
    for i, pose in enumerate(poses[: case["step"] + 1]):
        d.time = event.policy_start + i * DT
        d.qpos[:] = pose
        event.apply(sim)
        assert not any(
            block in (c.geom1, c.geom2) and (c.geom1 in robots or c.geom2 in robots) and c.dist <= 0
            for c in d.contact
        )

    # Try yielding on the original approach and resuming the same joint path
    # after the cylinder stops. This checks geometry, independent of any policy.
    def touching():
        return any(
            block in (c.geom1, c.geom2) and (c.geom1 in robots or c.geom2 in robots) and c.dist <= 0
            for c in d.contact
        )

    d.mocap_pos[event.mid] = event.end
    last_blocked = -1
    for i in range(len(poses) - 1, -1, -1):
        d.qpos[:] = poses[i]
        sim.forward()
        if touching():
            last_blocked = i
            break
    pauses = {}
    for speed in SPEEDS:
        env.obstacle_speed_mps = speed
        env.set_init_state(state)
        event = env.env.level_three_motion
        first_step = round(first_hits[str(speed)] / DT)
        pauses[str(speed)] = None
        for i in range(last_blocked + 1, first_step):
            d.qpos[:] = poses[i]
            pos = event.position(i * DT)
            samples = int(np.ceil(np.linalg.norm(event.end - pos) / 0.001)) + 1
            for point in np.linspace(pos, event.end, samples):
                d.mocap_pos[event.mid] = point
                sim.forward()
                if touching():
                    break
            else:
                pauses[str(speed)] = round(i * DT, 4)
                break
    return first_hits, earlier_object_contacts, pauses


def run(args):
    from libero.libero.envs.timed_crossing import timed_cases

    cls = benchmark.get_benchmark_dict()["safelibero_" + args.suite]
    suite = cls(safety_level="I" if args.mode == "build" else "III")
    source = suite.get_task_bddl_file_path(args.task)
    task = suite.get_task(args.task)
    states = suite.get_task_init_states(args.task)
    policy = None
    if args.mode in ("evaluate", "wait-control"):
        import dataclasses
        import torch
        from openpi.training import config
        from openpi.policies import policy_config
        from eval_paired_pi05 import rollout, LIMITS

        torch.set_num_threads(1)
        cfg = config.get_config("pi05_libero")
        cfg = dataclasses.replace(
            cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None)
        )
        policy = policy_config.create_trained_policy(cfg, args.checkpoint, pytorch_device="cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    traces = np.load(DOCS / "paired-pi05-traces.npz")
    trials = [
        (ep, v)
        for ep in ([0] if args.mode == "render" else args.episodes)
        for v in (args.speeds if args.mode == "evaluate" else [0.10])
    ]
    for ep, speed in trials:
        stem = f"{args.suite}-{args.task}-{ep}"
        suffix = f"-v{round(speed * 100)}" if args.mode == "evaluate" else ""
        path = args.output / (stem + suffix + ".json")
        if path.is_file() and args.mode != "render":
            continue
        state = states[ep]
        key = state_key(source, state)
        poses, xyz = traces[stem + "_qpos"], traces[stem + "_xyz"]
        random.seed(7 + ep)
        np.random.seed(7 + ep)
        env = make_preview_env(
            bddl_file_name=source,
            safety_level="III",
            hard_reset=False,
            use_camera_obs=policy is not None,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=256,
            camera_widths=256,
            control_freq=20,
            obstacle_speed_mps=speed,
        )
        try:
            env.reset()
            if args.mode == "build":
                case = build(env, state, poses, xyz, frozen_cases()[key])
                env.paired_case = case
            else:
                case = timed_cases()[key]
            if args.mode == "verify":
                check = lane_checker(env, state, case)
                assert check(np.array(case["start"]), np.array(case["end"])), (
                    "Frozen lane lacks reset clearance"
                )
            if args.mode in ("build", "verify"):
                first, others, pauses = verify(env, state, case, poses)
                row = dict(
                    case=case,
                    first_hits_s=first,
                    earlier_object_contacts=others,
                    pause_until_exit_s=pauses,
                )
            elif args.mode in ("evaluate", "wait-control"):
                wait_steps = 0
                if args.mode == "wait-control":
                    distance = np.linalg.norm(np.subtract(case["end"], case["center"]))
                    wait_steps = int(np.ceil((case["step"] * DT + distance / speed) / DT)) + 1
                with torch.inference_mode():
                    metrics, q, _, _ = rollout(
                        env,
                        state,
                        task.language,
                        policy,
                        LIMITS[args.suite],
                        7 + ep,
                        pause_steps=wait_steps,
                    )
                np.testing.assert_allclose(q[0], poses[0], atol=1e-8, rtol=0)
                event = env.env.level_three_motion
                stamp = env.sim._paired_first_added_sim_time
                t = None if stamp is None else stamp - event.policy_start
                assert (t is not None) == metrics["contacts"]["added"]
                phase = (
                    None
                    if t is None
                    else (
                        "before"
                        if t < event.onset
                        else "moving"
                        if t < event.onset + event.length / speed
                        else "stopped"
                    )
                )
                row = dict(
                    metrics=metrics,
                    speed_mps=speed,
                    first_added_s=t,
                    first_added_phase=phase,
                    initial_hold_steps=wait_steps,
                )
            else:
                from render_timed_crossing import render

                render(env, state, case, poses, xyz, args.output / f"{args.suite}-{args.task}")
                continue
        finally:
            env.close()
        row.update(suite=args.suite, task=args.task, episode=ep)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(row, indent=2))
        tmp.replace(path)
        print(args.mode, stem, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["build", "verify", "evaluate", "render", "wait-control"])
    p.add_argument("--suite", choices=["spatial", "object", "goal", "long"], required=True)
    p.add_argument("--task", type=int, choices=range(4), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, choices=range(3), nargs="+", default=[0, 1, 2])
    p.add_argument("--checkpoint", help="Converted OpenPI pi05_libero checkpoint")
    p.add_argument("--speeds", type=float, choices=SPEEDS, nargs="+", default=list(SPEEDS))
    args = p.parse_args()
    if args.mode in ("evaluate", "wait-control") and not args.checkpoint:
        p.error("evaluation requires --checkpoint")
    run(args)
