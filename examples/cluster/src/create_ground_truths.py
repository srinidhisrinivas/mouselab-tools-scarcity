from mouselab.mouselab import MouselabEnv
from mouselab.distributions import sample
import numpy as np
import sys
import random
import json
from pathlib import Path

from mouselab.envs.registry import register
from mouselab.distributions import Categorical

try:
    experiment_setting = sys.argv[1]
except:
    experiment_setting = 'high_increasing'

try:
    number = int(sys.argv[2])
except:
    number = 100

small_decreasing_reward = {
    1: Categorical([-48, -24, 24, 48]),
    2: Categorical([-8, -4, 4, 8]),
}

# create narrow_large_increasing
register(
    name="narrow_small_decreasing",
    branching=[1,2],
    reward_inputs=["depth"],
    reward_dictionary=small_decreasing_reward,
)

medium_decreasing_reward = {
    1: Categorical([-48, -24, 24, 48]),
    2: Categorical([-8, -4, 4, 8]),
    3: Categorical([-4, -2, 2, 4]),
}

medium_increasing_reward = {
    3: Categorical([-48, -24, 24, 48]),
    2: Categorical([-8, -4, 4, 8]),
    1: Categorical([-4, -2, 2, 4]),
}

# create narrow_large_increasing
register(
    name="medium_increasing",
    branching=[1,2,2],
    reward_inputs=["depth"],
    reward_dictionary=medium_increasing_reward,
)

# create narrow_large_increasing
register(
    name="medium_decreasing",
    branching=[1,2,2],
    reward_inputs=["depth"],
    reward_dictionary=medium_decreasing_reward,
)

env_increasing = MouselabEnv.new_symmetric_registered(experiment_setting)

outpath = Path(__file__).resolve().parents[1].joinpath(f"exp_inputs/rewards/g_truths.json")

glist = []
for i in range(number):
    sampled = (list(map(sample, env_increasing.init)))
    glist.append({
        "trial_id" : random.randint(100000000000, 999999999999),
        "stateRewards" : [float(s) for s in sampled]
    })

with open(outpath, 'w') as f:
    json.dump(glist, f)
