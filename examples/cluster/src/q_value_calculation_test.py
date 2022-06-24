import sys
from pathlib import Path

import dill as pickle

from mouselab.envs.registry import registry
import random

#print(registry)

from mouselab.envs.registry import register
from mouselab.distributions import Categorical


from mouselab.env_utils import get_ground_truths_from_json
from mouselab.exact_utils import timed_solve_env
from mouselab.mouselab import MouselabEnv

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


experiment_setting = "narrow_small_decreasing"
ground_truth_file = "./exp_inputs/rewards/12_24_4.json"
ground_truth_file = None

# make folder we need
Path(__file__).resolve().parents[1].joinpath("output").mkdir(
    parents=True, exist_ok=True
)

save_pi = True
save_q = False

if ground_truth_file is not None:
    states = get_ground_truths_from_json(ground_truth_file)
else:
    states = None

percent_rewarded = 1.0

env_increasing = MouselabEnv.new_symmetric_registered(experiment_setting)
env_increasing._pct_reward = percent_rewarded;

q, v, pi, info = timed_solve_env(env_increasing, save_pi=save_pi, save_q=save_q, ground_truths=states)
file_prefix = "example_q_dict" if save_q else "example_pi_dict"

percent_rewarded = 0.5

env_increasing = MouselabEnv.new_symmetric_registered(experiment_setting)
env_increasing._pct_reward = percent_rewarded;

q_2, v_2, pi_2, info_2 = timed_solve_env(env_increasing, save_pi=save_pi, save_q=save_q, ground_truths=states)


file_prefix = "example_q_dict" if save_q else "example_pi_dict"

num_equal = 0
num_diff = 0
num_sub = 0
num_int = 0

for state, actions_2 in info_2["pi_dictionary"].items():
    actions_1 = info["pi_dictionary"][state]
    if set(actions_1) == set(actions_2):
        num_equal += 1
        print("Equal")
        print(state)
        print(pi(state,print_Qs=True))
        print(pi_2(state,print_Qs=True))
        print("\n")
    elif set(actions_2).issubset(set(actions_1)):
        num_sub += 1
        print("Subset")
        print(state)
        print(pi(state,print_Qs=True))
        print(pi_2(state,print_Qs=True))
        print("\n")
    elif len(set(actions_1).intersection(set(actions_2))):
        num_int += 1
        print("Intersection")
        print(state)
        print(pi(state,print_Qs=True))
        print(pi_2(state,print_Qs=True))
        print("\n")
    else:
        num_diff += 1
        print("Difference")
        print(state)
        print(pi(state,print_Qs=True))
        print(pi_2(state,print_Qs=True))
        print("\n")


print("Number of similar actions: {}".format(num_equal))
print("Number of subset actions: {}".format(num_sub))
print("Number of intersection actions: {}".format(num_int))
print("Number of differing actions: {}".format(num_diff))
#
# path = (
#     Path(__file__)
#     .resolve()
#     .parents[1]
#     .joinpath(f"output/{file_prefix}_{experiment_setting}.pickle")
# )
#
# with open(path, "wb") as f:
#     pickle.dump(info, f)
#
# with open(path, "rb") as f:
#     loaded = pickle.load(f)
#     print(loaded)
#     if save_pi:
#         print(len(loaded["pi_dictionary"].items()));
#         print(loaded["pi_dictionary"].__sizeof__())
#     if save_q:
#         print(len(loaded["q_dictionary"].items()));
#         print(loaded["q_dictionary"].__sizeof__())