"""Export one annotated representative episode for a static SafeLIBERO task."""
import argparse
import json
from pathlib import Path
from libero.libero import benchmark
from scene_regions import SceneRegions, make_preview_env


def render(suite, level, task, out):
    suite_obj = benchmark.get_benchmark_dict()['safelibero_'+suite](safety_level=level)
    env = make_preview_env(bddl_file_name=suite_obj.get_task_bddl_file_path(task),
                             safety_level=level,use_camera_obs=False,hard_reset=False)
    try:
        env.seed(13)
        env.reset()
        for i,state in enumerate(suite_obj.get_task_init_states(task)):
            env.set_init_state(state)
            regions = SceneRegions(env)
            image = regions.frame(400)
            if min(regions.seen) >= 20:
                break
        else:
            raise ValueError('No episode with both roles visible')
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out, quality=89)
        out.with_suffix('.json').write_text(json.dumps(dict(suite=suite,task=task,level=level,
            name=suite_obj.get_task(task).name,state=i,visible_pixels=regions.seen.tolist())))
    finally:
        env.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',required=True)
    p.add_argument('--task',required=True,type=int)
    p.add_argument('--level',choices=['I','II'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    render(a.suite,a.level,a.task,a.output)
