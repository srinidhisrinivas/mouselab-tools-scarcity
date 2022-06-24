import sys
import dill as pickle
from pathlib import Path

try:
    experiment_setting = sys.argv[1]
except:
    raise "Experiment setting not present"

try:
    reward_pct_1 = sys.argv[2]
except:
    raise "Reward pctg 1 not present"

try:
    reward_pct_2 = sys.argv[3]
except:
    raise "Reward pctg 2 not present"
        
path_1 = (
    Path(__file__)
    .resolve()
    .parents[1]
    .joinpath(f"output/example_pi_dict_{experiment_setting}_{reward_pct_1}.pickle")
)

path_2 = (
    Path(__file__)
    .resolve()
    .parents[1]
    .joinpath(f"output/example_pi_dict_{experiment_setting}_{reward_pct_2}.pickle")
)

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
diff_dict = {}

for state, actions_2 in dict2.items():
    actions_1 = dict1[state]
    if set(actions_1) == set(actions_2):
        num_equal += 1
    else:
        num_diff += 1
        diff_dict[state] = { "1": actions_1, "2": actions_2 }

print("Number of similar actions: {}".format(num_equal))
print("Number of differing actions: {}".format(num_diff))

if len(diff_dict) > 0:
    diff_path = (
        Path(__file__)
            .resolve()
            .parents[1]
            .joinpath(f"output/diff_{experiment_setting}_{reward_pct_1}_{reward_pct_2}.pickle")
        )
    with open(diff_path, 'wb') as f:
        pickle.dump(diff_dict, f)