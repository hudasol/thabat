import sys, time
sys.path.insert(0, "../../src")
from thabat_study.tasks import ARMS, load_cfg, runner_for
from thabat_kit.seeding import seed_everything
for task, steps in [("pendulum", 1500), ("invpend", 1500), ("cartpole", 1500), ("grok", 300)]:
    cfg = load_cfg(task, need_budget=False); cfg["steps"] = steps
    if task == "grok": cfg["train_frac"] = 0.5
    fn = runner_for(task)
    res = {}
    for arm in ("ref", "ref_repeat", "thr", "fp64"):
        env = ARMS[arm]; env.apply()
        t0 = time.time(); r = fn(0, env, cfg); dt = time.time() - t0
        res[arm] = r; print(task, arm, round(dt, 1), "s", r["outcome"], round(r["score"], 3), r["param_hash"])
    print("  determinism:", res["ref"]["param_hash"] == res["ref_repeat"]["param_hash"],
          "| thr differs:", res["ref"]["param_hash"] != res["thr"]["param_hash"],
          "| fp64 differs:", res["ref"]["param_hash"] != res["fp64"]["param_hash"])
