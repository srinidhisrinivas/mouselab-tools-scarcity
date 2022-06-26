import sys
import dill as pickle
from pathlib import Path
import random

from mouselab.env_utils import get_ground_truths_from_json
from mouselab.exact_utils import timed_solve_env
from mouselab.mouselab import MouselabEnv

try:
    experiment_setting = sys.argv[1]
except:
    raise "Experiment setting not present"

try:
    reward_pct_1 = int(sys.argv[2])
    if reward_pct_1 < 0.50:
        raise "Too low"
except:
    raise "Reward pctg must be number between 0.50 and 1.00"

try:
    num_rewarded_trials = int(sys.argv[3])
    if num_rewarded_trials > 200:
        raise "Too many trials"
except:
    raise "Num rewarded trials must be number < 200"

try:
    ground_truth_file = sys.argv[3]
except:
    raise "Ground truth file not present"


ground_truths = get_ground_truths_from_json(ground_truth_file)

policy_file_name = (
    Path(__file__)
        .resolve()
        .parents[1]
        .joinpath(f"output/pi_dict_{experiment_setting}_{reward_pct_1}.pickle")
)

num_total_trials = round(num_rewarded_trials / reward_pct_1)

num_unrewarded_trials = num_total_trials - num_rewarded_trials

random.shuffle(ground_truths)

rewarded_trials = ground_truths[0:num_rewarded_trials]
assert len(rewarded_trials) == num_rewarded_trials

unrewarded_trials = ground_truths[num_unrewarded_trials:num_total_trials]
assert len(unrewarded_trials) == num_unrewarded_trials

