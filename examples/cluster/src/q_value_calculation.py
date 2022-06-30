import sys
from pathlib import Path

import dill as pickle

from mouselab.env_utils import get_ground_truths_from_json
from mouselab.exact_utils import timed_solve_env
from mouselab.mouselab import MouselabEnv

experiment_setting = sys.argv[1]
try:
    ground_truth_file = sys.argv[2]
except:
    ground_truth_file = None

try:
    percent_rewarded = float(sys.argv[3])
except:
    percent_rewarded = 1

try:
    save = sys.argv[4]
    if save == "q":
        save_pi = False
        save_q = True
    elif save == "pi":
        save_pi = True
        save_q = False
except:
    save_pi = True
    save_q = False

print("Experiment setting: {}".format(experiment_setting))
# make folder we need
Path(__file__).resolve().parents[1].joinpath("output").mkdir(
    parents=True, exist_ok=True
)
base_cost = 1
if ground_truth_file is not None:
    states = get_ground_truths_from_json(ground_truth_file)
else:
    states = None
env_increasing = MouselabEnv.new_symmetric_registered(experiment_setting, cost=base_cost * percent_rewarded)
env_increasing._pct_reward = percent_rewarded
q, v, pi, info = timed_solve_env(env_increasing, save_pi=save_pi, save_q=save_q, ground_truths=states, verbose=True)

file_prefix = "q_dict" if save_q else "pi_dict"

path = (
    Path(__file__)
    .resolve()
    .parents[1]
    .joinpath(f"output/{file_prefix}_{experiment_setting}_{percent_rewarded}.pickle")
)

with open(path, "wb") as f:
    pickle.dump(info, f)
