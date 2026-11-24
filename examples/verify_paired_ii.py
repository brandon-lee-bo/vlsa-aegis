"""Check the published Level II cases against the recorded Level I joint positions."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from libero.libero import benchmark
from libero.libero.envs.paired_obstacle import CATALOG, NAME, robot_geoms, state_key
from scene_regions import make_preview_env, paired_preview


def verify(suites, gallery):
    docs = Path(__file__).resolve().parents[1]/"docs"
    cases = json.loads(CATALOG.read_text())["cases"]
    traces = np.load(docs/"paired-pi05-traces.npz")
    verified = 0
    for suite_name in suites:
        klass = benchmark.get_benchmark_dict()["safelibero_"+suite_name]
        first, second = klass(safety_level="I"), klass(safety_level="II")
        for task in range(first.n_tasks):
            source = first.get_task_bddl_file_path(task)
            assert source == second.get_task_bddl_file_path(task)
            a,b = first.get_task_init_states(task),second.get_task_init_states(task)
            assert len(b) == 3
            env = make_preview_env(bddl_file_name=source,safety_level="II",
                                   hard_reset=False,use_camera_obs=False)
            try:
                for ep in (0,1,2):
                    np.testing.assert_array_equal(a[ep],b[ep])
                    env.set_init_state(b[ep])
                    case = cases[state_key(source,b[ep])]
                    sim = env.sim
                    geom = sim.model.geom_name2id(NAME)
                    data = sim.data._data
                    np.testing.assert_allclose(data.geom_xpos[geom],case["center"],atol=1e-9)
                    assert not any(geom in (c.geom1,c.geom2) and c.dist < -1e-5 for c in data.contact)
                    key = f"{suite_name}-{task}-{ep}"
                    data.qpos[:] = traces[key+"_qpos"][case["reference_step"]]
                    mujoco.mj_forward(sim.model._model,data)
                    expected = sim.model.geom_name2id(case["geom"])
                    assert expected in robot_geoms(env)
                    depths = [-c.dist for c in data.contact if {c.geom1,c.geom2} == {geom,expected}]
                    assert depths and max(depths) >= .001
                    verified += 1
                if gallery:
                    gallery.mkdir(parents=True,exist_ok=True)
                    key = f"{suite_name}-{task}-0"
                    for level in ("I","II"):
                        display = make_preview_env(bddl_file_name=source,safety_level=level,
                                                   hard_reset=False,use_camera_obs=False)
                        try:
                            display.reset()
                            display.set_init_state(a[0])
                            for _ in range(10):
                                display.step([0.]*6+[-1.])
                            paired_preview(display,traces[key+"_xyz"],
                                           cases[state_key(source,a[0])] if level=="II" else None,
                                           gallery/f"{level}-{suite_name}-{task}.webp")
                        finally:
                            display.close()
            finally:
                env.close()
    return verified


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suites",nargs="+",default=["spatial","object","goal","long"])
    p.add_argument("--gallery",type=Path,help="Render paired first-observation images")
    args = p.parse_args()
    print("Verified pairs:",verify(args.suites,args.gallery))
