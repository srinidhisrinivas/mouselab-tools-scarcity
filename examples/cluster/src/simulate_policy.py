import sys
import dill as pickle
from pathlib import Path
import random
from copy import deepcopy

from mouselab.envs.registry import register
from mouselab.distributions import Categorical
import math

from mouselab.env_utils import get_ground_truths_from_json
from mouselab.mouselab import MouselabEnv

# --- Register test environments ---

# create narrow_small_decreasing
small_decreasing_reward = {
    1: Categorical([-48, -24, 24, 48]),
    2: Categorical([-8, -4, 4, 8]),
}

register(
    name="narrow_small_decreasing",
    branching=[1, 2],
    reward_inputs=["depth"],
    reward_dictionary=small_decreasing_reward,
)

# create medium_decreasing
medium_decreasing_reward = {
    1: Categorical([-48, -24, 24, 48]),
    2: Categorical([-8, -4, 4, 8]),
    3: Categorical([-4, -2, 2, 4]),
}

register(
    name="medium_decreasing",
    branching=[1, 2, 2],
    reward_inputs=["depth"],
    reward_dictionary=medium_decreasing_reward,
)

# create narrow_large_increasing
medium_increasing_reward = {
    3: Categorical([-48, -24, 24, 48]),
    2: Categorical([-8, -4, 4, 8]),
    1: Categorical([-4, -2, 2, 4]),
}

register(
    name="medium_increasing",
    branching=[1, 2, 2],
    reward_inputs=["depth"],
    reward_dictionary=medium_increasing_reward,
)

# Marking the nodes at respective levels for 1-2-2 increasing (medium_increasing) environment
#   and 3-1-2 (high_increasing) environment
level_nodes = {
    "medium_increasing": {
        "1": [1],
        "2": [2, 5],
        "3": [3, 4, 6, 7]
    },
    "high_increasing": {
        "1": [1, 5, 9],
        "2": [2, 6, 10],
        "3": [3, 4, 7, 8, 11, 12]
    }
}

# --- Defining Constants ---
outcome_template = {
    "rewarded": {
        "num_clicks": [],
        "click_level" : [],
        "scores": [],
        "term_rewards": [],
        "costs": []
    },
    "unrewarded": {
        "num_clicks": [],
        "click_level" : [],
        "scores": [],
        "term_rewards": [],
        "costs": []
    }
}
base_cost = 1

# --- Reading input arguments ---
try:
    experiment_setting = sys.argv[1]
except:
    raise "Experiment setting not present"

try:
    ground_truth_file = sys.argv[2]
except:
    raise "Ground truth file not present"

try:
    num_rewarded_trials = int(sys.argv[3])
except:
    raise "Num rewarded trials must be number"

try:
    reward_pct_1 = float(sys.argv[4])
    if reward_pct_1 < 0.0:
        raise "Too low"
except:
    raise "Reward pctg must be number between 0.0 and 1.00"


# --- Building trials for simulation ---

# Assemble all the trials based on the scarcity level
ground_truths = get_ground_truths_from_json(ground_truth_file)
num_total_trials = round(num_rewarded_trials / reward_pct_1)

if num_total_trials > len(ground_truths):
    factor = num_total_trials / len(ground_truths)
else:
    factor = 1

ground_truths = ground_truths * math.ceil(factor)

num_unrewarded_trials = num_total_trials - num_rewarded_trials

random.shuffle(ground_truths)

rewarded_trials = ground_truths[0:num_rewarded_trials]
assert len(rewarded_trials) == num_rewarded_trials

unrewarded_trials = ground_truths[num_rewarded_trials:num_total_trials]
assert len(unrewarded_trials) == num_unrewarded_trials

trials_collected = {
    "rewarded": rewarded_trials,
    "unrewarded": unrewarded_trials
}

# --- Policies to be simulated ---

# Read the optimal policy for the given experiment_setting and environment
policy_file_name = (
    Path(__file__)
        .resolve()
        .parents[1]
        .joinpath(f"output/pi_dict_{experiment_setting}_{reward_pct_1}.pickle")
)

with open(policy_file_name, 'rb') as f:
    opt_policy = pickle.load(f)["pi_dictionary"]


# Return the action of the optimal policy
def optimal_policy(state):
    possible_actions = opt_policy[state]["max_actions"]
    return random.choice(possible_actions)

# Policy that clicks all the nodes at a given level in order and then terminates planning
def all_nodes_in_level_policy(state, exp_setting, level):
    desired_actions = level_nodes[exp_setting][str(level)]

    for a in desired_actions:
        try:
            node_val = int(state[a])
        except:
            return a

    # Terminate action if no unclicked node at the desired level found
    return len(state)


policy_functions = {
    "optimal": lambda state, exp_setting: optimal_policy(state),
    "level-1": lambda state, exp_setting: all_nodes_in_level_policy(state, exp_setting, "1"),
    "level-2": lambda state, exp_setting: all_nodes_in_level_policy(state, exp_setting, "2"),
    "level-3": lambda state, exp_setting: all_nodes_in_level_policy(state, exp_setting, "3")
}

#--- Running the simulation for all policies ---
policies_to_simulate = ["optimal", "level-1", "level-2", "level-3"]  # optimal, level-1, level-2, level-3
simulation_trial_outcomes = {policy: deepcopy(outcome_template) for policy in policies_to_simulate}

save_click_sequences_of = ["optimal"]

click_sequences = { policy: [] for policy in save_click_sequences_of }

for policy in policies_to_simulate:
    print("Simulating policy: {}".format(policy))
    p_func = policy_functions[policy]
    for trial_type, trials in trials_collected.items():
        print("\t{}: {} trials".format(trial_type, len(trials)))
        for trial in trials:
            env = MouselabEnv.new_symmetric_registered(
                    experiment_setting, ground_truth=trial, cost=base_cost * reward_pct_1,
                    term_belief=False, sample_term_reward=True
            )
            clicks_made = []
            env._is_scarce = True if trial_type == "unrewarded" else False
            current_state = env.init
            trial_score = 0
            trial_reward = 0
            trial_cost = 0
            num_clicks = 0
            click_level_tot = 0
            while True:
                action = p_func(current_state, experiment_setting)
                if action in level_nodes[experiment_setting]["1"]:
                    click_level_tot += 1
                elif action in level_nodes[experiment_setting]["2"]:
                    click_level_tot += 2
                elif action in level_nodes[experiment_setting]["3"]:
                    click_level_tot += 3

                clicks_made.append(action)
                current_state, reward, done, _ = env.step(action)
                if done:
                    trial_reward += reward
                    trial_score += reward
                    simulation_trial_outcomes[policy][trial_type]["scores"].append(trial_score)
                    simulation_trial_outcomes[policy][trial_type]["term_rewards"].append(trial_reward)
                    simulation_trial_outcomes[policy][trial_type]["costs"].append(trial_cost)
                    simulation_trial_outcomes[policy][trial_type]["num_clicks"].append(num_clicks)
                    if num_clicks == 0:
                        simulation_trial_outcomes[policy][trial_type]["click_level"].append(0)
                    else:
                        simulation_trial_outcomes[policy][trial_type]["click_level"].append(
                            (click_level_tot / num_clicks)
                        )
                    if policy in save_click_sequences_of:
                        click_sequences[policy].append({
                            "stateRewards": trial,
                            "clicks": clicks_made
                        })
                    break
                else:
                    num_clicks += 1
                    trial_cost += reward
                    trial_score += reward


# Combine results of rewarded and unrewarded
combined_results = {}
for policy, outcomes in simulation_trial_outcomes.items():
    combined = {}
    for measure in outcomes["rewarded"]:
        combined[measure] = outcomes["rewarded"][measure] + outcomes["unrewarded"][measure]
    combined_results[policy] = combined

for policy, combined_result in combined_results.items():
    simulation_trial_outcomes[policy]["combined"] = combined_result


# Compute benefit of 1st policy in list against other policies
# Benefit is ratio of the expected trial scores of both policies
baseline_policy = policies_to_simulate[0]
overall_benefits = {}
per_trial_benefits = {}
for compare_policy in policies_to_simulate[1:]:
    baseline_trial_scores = simulation_trial_outcomes[baseline_policy]["combined"]["scores"]
    compare_trial_scores = simulation_trial_outcomes[compare_policy]["combined"]["scores"]

    per_trial_benefit_factors = [baseline / compare for (baseline, compare) in zip(baseline_trial_scores, compare_trial_scores)]
    overall_benefit_factor = sum(baseline_trial_scores) / sum(compare_trial_scores)
    overall_benefits[compare_policy] = overall_benefit_factor
    per_trial_benefits[compare_policy] = sum(per_trial_benefit_factors)/len(per_trial_benefit_factors)

# --- Display Results of Simulations ---
print("--- Simulation Results: {} ---".format(reward_pct_1))
print("\n")
for policy, outcomes in simulation_trial_outcomes.items():
    print("Policy: {}\n".format(policy))
    print("\tRewarded trials:\t\t {}".format(len(trials_collected["rewarded"])))
    print("\t\tAverage term reward:\t {0:0.3f}".format(
        sum(outcomes["rewarded"]["term_rewards"]) / len(outcomes["rewarded"]["term_rewards"])))
    print("\t\tAverage costs:\t\t {0:0.3f}".format(
        sum(outcomes["rewarded"]["costs"]) / len(outcomes["rewarded"]["costs"])))
    print("\t\tTotal term reward:\t {0:0.3f}".format(
        sum(outcomes["rewarded"]["term_rewards"])))
    print("\t\tTotal costs:\t\t {0:0.3f}".format(sum(
        outcomes["rewarded"]["costs"])))
    print("\t\tAverage num clicks:\t {0:0.3f}".format(
        sum(outcomes["rewarded"]["num_clicks"]) / len(outcomes["rewarded"]["num_clicks"])))
    print("\t\tAverage click level:\t {0:0.3f}".format(
        sum(outcomes["rewarded"]["click_level"]) / len(outcomes["rewarded"]["click_level"])))
    print("\t\tAverage score:\t\t {0:0.3f}".format(
        sum(outcomes["rewarded"]["scores"]) / len(outcomes["rewarded"]["scores"])))
    print("\t\tTotal score:\t\t {0:0.3f}".format(
        sum(outcomes["rewarded"]["scores"])))
    print("\n")

    if len(unrewarded_trials) > 0:
        print("\tUnrewarded trials:\t\t {}".format(len(trials_collected["unrewarded"])))
        print("\t\tAverage term reward:\t {0:0.3f}".format(
            sum(outcomes["unrewarded"]["term_rewards"]) / len(outcomes["unrewarded"]["term_rewards"])))
        print("\t\tAverage costs:\t\t {0:0.3f}".format(
            sum(outcomes["unrewarded"]["costs"]) / len(outcomes["unrewarded"]["costs"])))
        print("\t\tTotal term reward:\t {0:0.3f}".format(
            sum(outcomes["unrewarded"]["term_rewards"])))
        print("\t\tTotal costs:\t\t {0:0.3f}".format(sum(
            outcomes["unrewarded"]["costs"])))
        print("\t\tAverage num clicks:\t {0:0.3f}".format(
            sum(outcomes["unrewarded"]["num_clicks"]) / len(outcomes["unrewarded"]["num_clicks"])))
        print("\t\tAverage click level:\t {0:0.3f}".format(
            sum(outcomes["unrewarded"]["click_level"]) / len(outcomes["unrewarded"]["click_level"])))
        print("\t\tAverage score:\t\t {0:0.3f}".format(
            sum(outcomes["unrewarded"]["scores"]) / len(outcomes["unrewarded"]["scores"])))
        print("\t\tTotal score:\t\t {0:0.3f}".format(
            sum(outcomes["unrewarded"]["scores"])))
        print("\n")

    print("\tTotal trials:\t\t\t {}".format(len(trials_collected["rewarded"] + trials_collected["unrewarded"])))
    print("\t\tAverage term reward:\t {0:0.3f}".format(
        sum(outcomes["combined"]["term_rewards"]) / len(outcomes["combined"]["term_rewards"])))
    print("\t\tAverage costs:\t\t {0:0.3f}".format(
        sum(outcomes["combined"]["costs"]) / len(outcomes["combined"]["costs"])))
    print("\t\tTotal term reward:\t {0:0.3f}".format(
        sum(outcomes["combined"]["term_rewards"])))
    print("\t\tTotal costs:\t\t {0:0.3f}".format(sum(
        outcomes["combined"]["costs"])))
    print("\t\tAverage num clicks:\t {0:0.3f}".format(
        sum(outcomes["combined"]["num_clicks"]) / len(outcomes["combined"]["num_clicks"])))
    print("\t\tAverage click level:\t {0:0.3f}".format(
        sum(outcomes["combined"]["click_level"]) / len(outcomes["combined"]["click_level"])))
    print("\t\tAverage score:\t\t {0:0.3f}".format(
        sum(outcomes["combined"]["scores"]) / len(outcomes["combined"]["scores"])))
    print("\t\tTotal score:\t\t {0:0.3f}".format(
        sum(outcomes["combined"]["scores"])))
    print("\n")

if len(overall_benefits) > 0:
    print("--- Expected overall Benefit ---")
    print("\n")
    print("Benefit of policy {} in comparison to:\n".format(baseline_policy))
    for policy, factor in overall_benefits.items():
        print("\t{0}: {1:0.4f}".format(policy, factor))

if len(per_trial_benefits) > 0:
    print("--- Expected per trial Benefit ---")
    print("\n")
    print("Benefit of policy {} in comparison to:\n".format(baseline_policy))
    for policy, factor in per_trial_benefits.items():
        print("\t{0}: {1:0.4f}".format(policy, factor))



path = (
    Path(__file__)
        .resolve()
        .parents[1]
        .joinpath(f"output/{reward_pct_1}_clicks_{experiment_setting}_.pickle")
)

with open(path, "wb") as f:
    pickle.dump(click_sequences, f)