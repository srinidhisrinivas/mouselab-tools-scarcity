from contexttimer import Timer

from mouselab.env_utils import (
    get_all_possible_sa_pairs_for_env,
    get_all_possible_states_for_ground_truths,
    get_sa_pairs_from_states,
    get_all_possible_states_for_env,
    deduplicate_states
)
from mouselab.exact import solve

import gc
import resource

def timed_solve_env(
    env, verbose=True, save_q=False, save_pi=False, ground_truths=None, **solve_kwargs
):
    """
    Solves environment, saves elapsed time and optionally prints value and elapsed time
    :param env: MouselabEnv with only discrete distribution (must not be too big)
    :param verbose: Whether or not to print out solve information once done
    :return: Q, V, pi, info
             Q, V, pi are all recursive functions
             info contains the number of times Q and V were called
                as well as the elapsed time ("time")
    """
    with Timer() as t:
        Q, V, pi, info = solve(env, **solve_kwargs)
        info["time"] = t.elapsed
        if verbose:
            optimal_value = sum(
                V(s)
                for s, p in zip(env.initial_states, env.initial_state_probabilities)
            )
            print("optimal -> {:.2f} in {:.3f} sec".format(optimal_value, t.elapsed))
        elif (save_q or save_pi):
            # call V to cache q_dictionary
            for s in env.initial_states:
                V(s)

        #  Save Q function or Pi function
        if ground_truths is not None:
            # In some cases, it is too costly to save whole Q function
            if save_q:
                info["q_dictionary"] = construct_partial_q_dictionary(Q, env, ground_truths)
            elif save_pi:
                info["pi_dictionary"] = construct_partial_pi_dictionary(pi, env, ground_truths)
        else:
            if save_q:
                info["q_dictionary"] = construct_q_dictionary(Q, env, verbose)
            elif save_pi:
                info["pi_dictionary"] = construct_pi_dictionary(pi, env, verbose)

    return Q, V, pi, info

def construct_pi_dictionary(pi, env, verbose=False):
    """
    Construct pi dictionary for env, given environment is solved 
    """
    all_states = get_all_possible_states_for_env(env)
    dedup_states = deduplicate_states(
        all_states, verbose=verbose
    )
    states_gen = (s for s in dedup_states)

    print("States size: {}, Length: {}".format(dedup_states.__sizeof__(), len(dedup_states)));
    print("Generator size: {}".format(states_gen.__sizeof__()));
    del dedup_states
    gc.collect()
    pi_dict = {tuple(state): pi(tuple(state)) for state in states_gen}
    print("Pi Dict size: {}".format(pi_dict.__sizeof__()));
    return pi_dict

def construct_q_dictionary(Q, env, verbose=False):
    """
    Construct q dictionary for env, given environment is solved
    """

    sa = get_all_possible_sa_pairs_for_env(env, verbose=verbose)
    q_dictionary = {pair: Q(*pair) for pair in sa}
    return q_dictionary


def construct_partial_q_dictionary(Q, env, selected_ground_truths):
    """
    Construct q dictionary for only specified ground truth values
    """
    all_possible_states = get_all_possible_states_for_ground_truths(
        env, selected_ground_truths
    )
    sa = get_sa_pairs_from_states(all_possible_states)
    q_dictionary = {pair: Q(*pair) for pair in sa}
    return q_dictionary

def construct_partial_pi_dictionary(pi, env, selected_ground_truths):
    """
    Construct pi dictionary for only specified ground truth values
    """
    all_possible_states = get_all_possible_states_for_ground_truths(
        env, selected_ground_truths
    )
    pi_dictionary = {tuple(state): pi(tuple(state)) for state in all_possible_states}

    return pi_dictionary
