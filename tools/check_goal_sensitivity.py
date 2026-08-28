"""Diagnostic: does the goal cloud actually change the policy's output?

Usage:
    python tools/check_goal_sensitivity.py <checkpoint_dir>

Why this matters: if goal conditioning were silently disconnected, the training loss would still
look healthy -- the policy would just learn goal-averaged trajectories -- and the simulation
success rate would be poor for a reason no metric would point at. This compares the action under
four different goals against the action under a re-sampled but identical goal, so the goal-driven
change can be read against the nuisance floor.


Same observation, different goals -> the actions must differ, and differ *systematically*
(a goal pushed further right should not produce an identical trajectory). If they are identical
the goal is not wired into the network, and every success number would be meaningless.
"""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/home/samsung/3D_Bimanual_repo/scripts_isaaclab/irregular/pc_common")
import pc_ops
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors

CKPT = sys.argv[1]
cfg = PreTrainedConfig.from_pretrained(CKPT); cfg.pretrained_path = CKPT; cfg.device = "cpu"
ds = LeRobotDataset("local/push_pc1024", root="/home/samsung/data/push_pc1024")
policy = make_policy(cfg=cfg, ds_meta=ds.meta); policy.eval()
pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=CKPT,
                                     preprocessor_overrides={"device_processor": {"device": "cpu"}})

item = ds[0]
obs_pc = item["observation.point_cloud"].numpy()
state = item["observation.state"].numpy()
obj0 = pc_ops.isolate_object(obs_pc)
rng = np.random.default_rng(0)

def act(delta, seed):
    policy.reset()
    goal = pc_ops.make_goal_cloud(obj0, delta, cfg.goal_pc_feature.shape[0], np.random.default_rng(seed))
    batch = {"observation.state": torch.from_numpy(state)[None],
             "observation.point_cloud": torch.from_numpy(obs_pc)[None],
             "observation.goal_point_cloud": torch.from_numpy(goal)[None]}
    torch.manual_seed(0)          # same diffusion noise, so only the goal differs
    a = policy.select_action(pre(batch))
    return post(a).detach().numpy()[0]

print(f"object points used for goal cloud: {len(obj0)}")
goals = {"+x (0.15, 0.00)": (0.15, 0.0), "-x (-0.15, 0.00)": (-0.15, 0.0),
         "+y (0.00, 0.15)": (0.0, 0.15), "-y (0.00,-0.15)": (0.0, -0.15)}
acts = {k: act(d, 1) for k, d in goals.items()}
base = acts["+x (0.15, 0.00)"]
print("\naction (first 7 dims = left arm + gripper):")
for k, a in acts.items():
    print(f"  {k:18s} {np.round(a[:7], 4)}")
print("\nL2 distance from the +x goal's action:")
for k, a in acts.items():
    print(f"  {k:18s} {np.linalg.norm(a - base):.5f}")

# control: same goal, different resample seed -> should be much smaller than goal-driven change
same = np.linalg.norm(act((0.15, 0.0), 2) - base)
spread = max(np.linalg.norm(a - base) for a in acts.values())
print(f"\nsame goal, different point sampling : {same:.5f}   <- nuisance variation")
print(f"largest goal-driven change          : {spread:.5f}   <- signal")
print(f"\nsignal / nuisance ratio: {spread / max(same, 1e-9):.1f}x")
print("VERDICT:", "goal IS driving the policy" if spread > 5 * max(same, 1e-9)
      else "WARNING - goal has little or no effect on the output")
