from toolz import memoize


def sort_tree(env, state):
    """Breaks symmetry between belief states.

    This is done by enforcing that the knowledge about states at each
    depth be sorted by [0, 1, UNKNOWN]
    """
    state = list(state)
    for i in range(len(env.tree) - 1, -1, -1):
        if not env.tree[i]:
            continue
        c1, c2 = env.tree[i]
        idx1, idx2 = env.subtree_slices[c1], env.subtree_slices[c2]

        if not (state[idx1] <= state[idx2]):
            state[idx1], state[idx2] = state[idx2], state[idx1]
    return tuple(state)


def hash_tree(env, state):
    """Breaks symmetry between belief states."""
    if state == "__term_state__":
        return hash(state)

    def rec(n):
        x = hash(state[n])
        childs = sum(rec(c) for c in env.tree[n])
        return hash(str(x + childs))

    return rec(0)


def solve(env, hash_state=None, actions=None, blinkered=None):
    """Returns Q, V, pi, and computation data for an mdp environment."""
    info = {"q": 0, "v": 0}  # track number of times each function is called

    if hash_state is None:
        if hasattr(env, "n_arm"):
            hash_state = lambda state: tuple(sorted(state))
        elif hasattr(env, "tree"):
            # hash_state = lambda state: sort_tree(env, state)
            hash_state = lambda state: hash_tree(env, state)
    if actions is None:
        actions = env.actions
    if blinkered == "recursive":

        def subset_actions(a):
            if a == env.term_action:
                return ()
            return (*env.subtree[a][1:], *env.path_to(a)[:-1], env.term_action)

    elif blinkered == "children":

        def subset_actions(a):
            if a == env.term_action:
                return ()
            return (*env.subtree[a][1:], env.term_action)

    elif blinkered == "branch":
        assert hasattr(env, "_relevant_subtree")

        def subset_actions(a):
            if a == env.term_action:
                return ()
            else:
                return (*env._relevant_subtree(a), env.term_action)

    elif blinkered:

        def subset_actions(a):
            return (a, env.term_action)

    else:
        subset_actions = lambda a: None

    if hash_state is not None:

        def hash_key(args, kwargs):
            state = args[0]
            if state is None:
                return state
            else:
                if kwargs:
                    # Blinkered approximation. Hash key is insensitive
                    # to states that can't be acted on, except for the
                    # best expected value
                    # Embed the action subset into the state.
                    action_subset = kwargs["action_subset"]
                    mask = [0] * len(state)
                    for a in action_subset:
                        mask[a] = 1
                    state = tuple(zip(state, mask))
                return hash_state(state)

    else:
        hash_key = None

    def Q(s, a):
        info["q"] += 1
        action_subset = subset_actions(a)
        return round(sum(sp * (rp * r + V(s1, action_subset)) for sp, s1, r, rp in env.results(s, a)), 8)

    @memoize(key=hash_key)
    def V(s, action_subset=None):
        if s is None:
            return 0
        info["v"] += 1
        acts = actions(s)
        if action_subset is not None:
            acts = tuple(a for a in acts if a in action_subset)
        return max((Q(s, a) for a in acts), default=0)

    # Returns set of actions that yield the highest Q-value when in a given state
    def pi(s, print_Qs=False):
        diff_threshold = 0.0000001
        action_vals = {a: Q(s, a) for a in actions(s)}
        if print_Qs:
            print(action_vals)
        max_action_val = max(action_vals.values())
        return [k for k, v in action_vals.items() if abs(v - max_action_val) < diff_threshold], action_vals

    return Q, V, pi, info
