import joblib
import numpy as np
import pandas as pd
from toolz import partition_all

from mouselab.agents import Agent
from mouselab.envs.registry import registry
from mouselab.mouselab import MouselabEnv

def make_envs(cost=1.00, n=100, seed=None, env_type="constant_high"):
    if seed is not None:
        np.random.seed(seed)

    # get env details
    env_details = registry(env_type)

    # construct lst of envs using env details
    envs = [
        MouselabEnv.new_symmetric(
            env_details.branching, env_details.reward_function, cost=cost
        )
        for _ in range(n)
    ]

    return envs


def make_env(env_type, cost=1.0, seed=None, **kwargs):
    if seed is not None:
        np.random.seed(seed)

    # get env details
    env_details = registry(env_type)

    return MouselabEnv.new_symmetric(
        env_details.branching, env_details.reward_function, cost=cost, **kwargs
    )


def encode_state(state):
    return " ".join("_" if hasattr(x, "sample") else str(x) for x in state)


def evaluate(policy, envs):
    agent = Agent()

    def run_env(policy, env):
        agent.register(env)
        agent.register(policy)
        tr = agent.run_episode()
        return {"util": tr["return"], "observations": len(tr["actions"]) - 1}

    return pd.DataFrame(run_env(policy, env) for env in envs)


__ENVS = None
__AGENT = Agent()
__CHUNKS = None


def eval_chunk(i, return_mean=True):
    # Each process should start with a different random seed.
    np.random.seed(np.random.randint(1000) + i)
    returns = []
    for env in __CHUNKS[i]:
        __AGENT.register(env)
        returns.append(__AGENT.run_episode()["return"])
    if return_mean:
        return np.mean(returns)
    else:
        return returns


def get_util(policy, envs, parallel=None, return_mean=True):
    if parallel is None:
        util = evaluate(policy, envs).util
        if return_mean:
            return util.mean()
        else:
            return util
    else:
        np.random.randint(1000)  # cycle the random number generator
        global __CHUNKS
        chunk_size = len(envs) // parallel.n_jobs
        __CHUNKS = list(partition_all(chunk_size, envs))
        __AGENT.register(policy)
        jobs = (
            joblib.delayed(eval_chunk)(i, return_mean) for i in range(len(__CHUNKS))
        )
        result = parallel(jobs)
        if return_mean:
            return np.mean(result)
        else:
            return np.concatenate(result)
