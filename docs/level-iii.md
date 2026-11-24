# SafeLIBERO Level III · Moving obstacles

This R³ extension starts from SafeLIBERO Level I. Each of the 16 tasks keeps its instruction, goal predicate and 50 supplied initial states. One existing obstacle is placed on a reproducible 16 cm lane on the workspace surface. It waits, travels in a straight line and stops. The lane is determined at initialization and is independent of the policy and obstacle speed.

[Browse the 16 tasks and compare two speeds](https://brandon-lee-bo.github.io/VLSA-AEGIS.github.io/benchmark.html#level-iii) · [Original SafeLIBERO](https://github.com/THU-RCSCT/vlsa-aegis)

## Render one episode

After installing the upstream SafeLIBERO simulation dependencies and setting the LIBERO asset paths, run from the repository root. The preview uses Pillow, MuJoCo 3 (tested with 3.2.3), and the original init-state files. For PyTorch 2.6+ the upstream loader may require `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for those trusted files.

```sh
python examples/preview_level_iii.py --suite spatial --task 0 --state 0 \
    --speed 0.10 --delay 0.60 --output spatial.gif
```

Speed is in metres per second and delay is in simulated seconds. Use speed 0 for a stationary obstacle at the same start. The preview applies a zero-motion command, renders at 10 frames/s and writes the obstacle, waypoints and measured position error to a JSON file beside the GIF. The yellow ring marks the obstacle.

## Use with a policy

Select `--safety-level III` in the existing evaluation scripts. `--obstacle-speed-mps` and `--obstacle-start-s` control the event. Those entry points add the configured settling period to the delay, so motion starts relative to policy execution. For a custom evaluator, use the same level for the suite and the environment, then call `env.set_init_state(suite.get_task_init_states(i)[episode])`. At the environment API the delay starts at that call. The obstacle advances in simulated time inside the physics substeps of `env.step(action)`.

The generator uses collision bounds to select a supported, initially clear lane near the obstacle's supplied position. The moving obstacle follows a prescribed pose and can make contact with the robot. Speed changes keep the endpoints fixed. Whether a policy crosses the lane at the same time, collides, or completes the task requires a closed-loop rollout.

## Coverage

All 800 supplied Level I initial states produced a route. For each state we tested five positions along the lane against the initial scene at a 1 mm penetration threshold. For the first state of each task we also checked that changing the speed and delay preserved the endpoints. The web gallery shows state 0 for each task at 0.10 and 0.20 m/s.
