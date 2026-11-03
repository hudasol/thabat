import sys, time
sys.path.insert(0, "../../src")
from thabat_study.tasks import ARMS, load_cfg, runner_for
for hidden, batch in [(64,64),(128,128),(256,256)]:
    cfg = load_cfg("pendulum", need_budget=False); cfg.update(steps=2000, hidden=hidden, batch=batch)
    fn = runner_for("pendulum"); out = {}
    for arm in ("ref","thr"):
        env = ARMS[arm]; env.apply(); t0=time.time(); r = fn(0, env, cfg); out[arm]=(r["param_hash"], round(time.time()-t0,1))
    print(hidden, batch, "thr differs:", out["ref"][0]!=out["thr"][0], "times", out["ref"][1], out["thr"][1])
