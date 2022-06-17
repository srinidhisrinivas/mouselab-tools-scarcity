import sys
from pathlib import Path

import dill as pickle

from mouselab.env_utils import get_ground_truths_from_json
from mouselab.exact_utils import timed_solve_env
from mouselab.mouselab import MouselabEnv

experiment_setting = sys.argv[1]
print("Experiment setting: {}".format(experiment_setting))
# make folder we need
Path(__file__).resolve().parents[1].joinpath("output").mkdir(
    parents=True, exist_ok=True
)

save_pi = True
save_q = False

# states = get_ground_truths_from_json(ground_truth_file)
env_increasing = MouselabEnv.new_symmetric_registered(experiment_setting)
q, v, pi, info = timed_solve_env(env_increasing, save_pi=save_pi, save_q=save_q)

file_prefix = "example_q_dict" if save_q else "example_pi_dict"

path = (
    Path(__file__)
    .resolve()
    .parents[1]
    .joinpath(f"output/{file_prefix}_{experiment_setting}.pickle")
)

with open(path, "wb") as f:
    pickle.dump(info, f)