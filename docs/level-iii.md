# SafeLIBERO Level III · Moving obstacles

This R³ extension starts from SafeLIBERO Level I. Each of the 16 tasks keeps its instruction, goal predicate and 50 supplied initial states. In the linear event, one existing obstacle is placed on a reproducible 16 cm lane on the workspace surface. It waits, travels in a straight line and stops. The lane is determined at initialization and is independent of the policy and obstacle speed.

[Browse the 16 tasks and compare five speeds](https://brandon-lee-bo.github.io/VLSA-AEGIS.github.io/benchmark.html#level-iii) · [Original SafeLIBERO](https://github.com/THU-RCSCT/vlsa-aegis)

## Render one episode

After installing the upstream SafeLIBERO simulation dependencies and setting the LIBERO asset paths, run from the repository root. The preview uses Pillow, MuJoCo 3 (tested with 3.2.3), and the original init-state files. For PyTorch 2.6+ the upstream loader may require `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for those trusted files.

```sh
python examples/preview_level_iii.py --suite spatial --task 0 --cross-task-path \
    --speed 0.10 --delay 0.60 --output spatial.gif
```

Speed is in metres per second and delay is in simulated seconds. Use speed 0 for a stationary obstacle at the same start. The preview applies a zero-motion command, renders at 10 frames/s (20 frames/s for linear motion at 40 and 80 cm/s) and writes the obstacle, waypoints and measured position error to a JSON file beside the GIF. In crossing mode, yellow dashes connect the initial gripper pose, each pickup and its placement in instruction order. The drawer task includes reaching and opening the drawer. Two short bowl transfers include a lateral carry waypoint. The cyan line follows a point on the moving obstacle. A yellow ring appears on frames where its collision geometry intersects the indicated 3D segment. These are illustrative routes with a stationary arm. A policy may execute a different trajectory.

## Use with a policy

Select `--safety-level III` in the existing evaluation scripts. `--obstacle-speed-mps` and `--obstacle-start-s` control the event. Those entry points add the configured settling period to the delay, so motion starts relative to policy execution. For a custom evaluator, use the same level for the suite and the environment, then call `env.set_init_state(suite.get_task_init_states(i)[episode])`. At the environment API the delay starts at that call. The obstacle advances in simulated time inside the physics substeps of `env.step(action)`.

The generator uses collision bounds to select a supported, initially clear lane near the obstacle's supplied position. The moving obstacle follows a prescribed pose and can make contact with the robot. Speed changes keep the endpoints fixed. Whether a policy crosses the lane at the same time, collides, or completes the task requires a closed-loop rollout.

## Coverage

All 800 supplied Level I initial states produced a route. For each state we tested five positions along the lane against the initial scene at a 1 mm penetration threshold. For the first state of each task we also checked that changing the speed and delay preserved the endpoints. That audit covers the default lane generator. The website shows one representative episode per task and level. Levels I and II each load a single still of the selected scene. Level III runs the same episode at nominal obstacle speeds of 5, 10, 20, 40 and 80 cm/s. Changing speed preserves the initial state and event geometry. The blue and orange contours are derived from the visible MuJoCo geometry IDs of the manipulated objects and obstacles. Multi-object instructions mark both pickups. A moving hand and the object it carries share the obstacle color.

The [gallery episode manifest](gallery-episodes.json) lists the task index, initial state and event for each preview. Its crossing times are sampled in simulated seconds. These are selected demonstration episodes with a stationary robot. The [linear-lane records](level-iii-gallery.json) give the original lane coordinates and two-speed measurements.

## Reproduce a gallery episode

Use the level, task and initial state from the manifest. For an ordinary crossing, pass `cross_task_path=True` to the environment. For a carry-and-place episode, pass `hand_places_cup=True` instead. Both use `obstacle_speed_mps`. The carry preset needs an active red mug and a plate destination. The hand eases into and out of each movement, so the displayed rate is nominal. The mug moves freely after release. Robot contact with the hand or forearm and with the cup is reported separately in the step info.

```sh
python examples/preview_static_scene.py --suite goal --task 0 --level II --output goal-II.webp
python examples/preview_hand_cup.py --suite spatial --task 0 --state 4 --speed .40 --output spatial.gif
```
