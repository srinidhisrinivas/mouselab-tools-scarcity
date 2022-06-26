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

medium_decreasing_reward = {
    1: Categorical([-48, -24, 24, 48]),
    2: Categorical([-8, -4, 4, 8]),
    3: Categorical([-4, -2, 2, 4]),
}

# create narrow_large_increasing
register(
    name="medium_decreasing",
    branching=[1,2,2],
    reward_inputs=["depth"],
    reward_dictionary=medium_decreasing_reward,
)


experiment_setting = "medium_decreasing"
ground_truth_file = "./exp_inputs/rewards/122_24_4_2.json"
# ground_truth_file = None

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

percent_rewarded_1 = 1.0

env_increasing = MouselabEnv.new_symmetric_registered(experiment_setting)
env_increasing._pct_reward = percent_rewarded_1;

q, v, pi, info = timed_solve_env(env_increasing, save_pi=save_pi, save_q=save_q, ground_truths=states)
file_prefix = "q_dict" if save_q else "pi_dict"

percent_rewarded_2 = 0.7

env_increasing = MouselabEnv.new_symmetric_registered(experiment_setting)
env_increasing._pct_reward = percent_rewarded_2;

q_2, v_2, pi_2, info_2 = timed_solve_env(env_increasing, save_pi=save_pi, save_q=save_q, ground_truths=states)

num_equal = 0
num_diff = 0
num_sub = 0
num_int = 0

print("\nComparing directly")
for state, outcomes_2 in info_2["pi_dictionary"].items():
    outcomes_1 = info["pi_dictionary"][state]
    actions_1 = outcomes_1["max_actions"]
    actions_2 = outcomes_2["max_actions"]
    if set(actions_1) == set(actions_2):
        num_equal += 1
        # print("Equal")
        # print(state)
        # print(outcomes_1["q_values"])
        # print(outcomes_2["q_values"])
        # print("\n")
    elif set(actions_2).issubset(set(actions_1)):
        num_sub += 1
        print("Subset")
        print(state)
        print("1:")
        print(actions_1)
        print(outcomes_1["q_values"])
        print("2:")
        print(actions_2)
        print(outcomes_2["q_values"])
        print("\n")
    elif len(set(actions_1).intersection(set(actions_2))):
        num_int += 1
        print("Intersection")
        print(state)
        print("1:")
        print(actions_1)
        print(outcomes_1["q_values"])
        print("2:")
        print(actions_2)
        print(outcomes_2["q_values"])
        print("\n")
    else:
        num_diff += 1
        print("Difference")
        print(state)
        print("1:")
        print(actions_1)
        print(outcomes_1["q_values"])
        print("2:")
        print(actions_2)
        print(outcomes_2["q_values"])
        print("\n")

print("{} vs. {}".format(percent_rewarded_1, percent_rewarded_2))
print("Length of dict 1: {}".format(len(info["pi_dictionary"])))
print("Length of dict 2: {}".format(len(info_2["pi_dictionary"])))
print("Number of similar actions: {}".format(num_equal))
print("Number of subset actions: {}".format(num_sub))
print("Number of intersection actions: {}".format(num_int))
print("Number of differing actions: {}".format(num_diff))

path_1 = (
    Path(__file__)
    .resolve()
    .parents[1]
    .joinpath(f"output/{file_prefix}_{experiment_setting}_{percent_rewarded_1}.pickle")
)

with open(path_1, "wb") as f:
    pickle.dump(info, f)

path_2 = (
    Path(__file__)
        .resolve()
        .parents[1]
        .joinpath(f"output/{file_prefix}_{experiment_setting}_{percent_rewarded_2}.pickle")
)

with open(path_2, "wb") as f:
    pickle.dump(info_2, f)

print("\nComparing after reading file")
try:
    with open(path_1, "rb") as f:
        p1 = pickle.load(f)
        if "pi_dictionary" in p1:
            key = "pi_dictionary"
        else:
            key = "q_dictionary"
        dict1 = p1[key]
except:
    raise "File cannot be read: {}".format(path_1)


try:
    with open(path_2, "rb") as f:
        p2 = pickle.load(f)
        if "pi_dictionary" in p2:
            key = "pi_dictionary"
        else:
            key = "q_dictionary"
        dict2 = p2[key]
except:
    raise "File cannot be read: {}".format(path_1)

num_equal = 0
num_diff = 0
num_sub = 0
num_int = 0
diff_dict = {}
sub_dict = {}
int_dict = {}

for state, outcomes_2 in dict2.items():
    outcomes_1 = dict1[state]
    actions_1 = outcomes_1["max_actions"]
    actions_2 = outcomes_2["max_actions"]
    if set(actions_1) == set(actions_2):
        num_equal += 1
    elif set(actions_2).issubset(set(actions_1)):
        num_sub += 1
        sub_dict[state] = {
            "1": {
                "max_actions": actions_1,
                "q_values": outcomes_1["q_values"]
            },
            "2": {
                "max_actions": actions_2,
                "q_values": outcomes_2["q_values"]
            }
        }
    elif len(set(actions_1).intersection(set(actions_2))):
        num_int += 1
        int_dict[state] = {
            "1": {
                "max_actions": actions_1,
                "q_values": outcomes_1["q_values"]
            },
            "2": {
                "max_actions": actions_2,
                "q_values": outcomes_2["q_values"]
            }
        }
    else:
        num_diff += 1
        diff_dict[state] = {
            "1": {
                "max_actions": actions_1,
                "q_values": outcomes_1["q_values"]
            },
            "2": {
                "max_actions": actions_2,
                "q_values": outcomes_2["q_values"]
            }
        }

print("{} vs. {}".format(percent_rewarded_1, percent_rewarded_2))
print("Length of dict 1: {}".format(len(dict1)))
print("Length of dict 2: {}".format(len(dict2)))
print("Number of similar actions: {}".format(num_equal))
print("Number of subset actions: {}".format(num_sub))
print("Number of intersection actions: {}".format(num_int))
print("Number of differing actions: {}".format(num_diff))