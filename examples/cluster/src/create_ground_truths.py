from mouselab.mouselab import MouselabEnv
from mouselab.distributions import sample
import numpy as np
import sys
import random
import json
from pathlib import Path

try:
    number = int(sys.argv[1])
except:
    number = 100

experiment_setting = "high_increasing"
env_increasing = MouselabEnv.new_symmetric_registered(experiment_setting)

outpath = Path(__file__).resolve().parents[1].joinpath(f"exp_inputs/rewards/g_truths.json")

glist = []
for i in range(number):
    sampled = (list(map(sample, env_increasing.init)))
    glist.append({
        "trialId" : random.randint(100000000000, 999999999999),
        "stateRewards" : [float(s) for s in sampled]
    })

with open(outpath, 'w') as f:
    json.dump(glist, f)
