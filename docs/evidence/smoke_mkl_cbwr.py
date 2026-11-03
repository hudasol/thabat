import sys
sys.path.insert(0, "../../src")
from thabat_study.tasks import ARMS, load_cfg, runner_for
import torch
cfg = load_cfg("pendulum", need_budget=False); cfg.update(steps=2000)
env = ARMS["ref"]; env.apply(); r = runner_for("pendulum")(0, env, cfg)
print(r["param_hash"], round(r["score"],2), torch.backends.mkl.is_available())
