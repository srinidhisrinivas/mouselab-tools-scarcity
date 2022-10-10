from mouselab.mouselab import MouselabEnv

from mouselab.distributions import PointMass, cmax, expectation, sample, smax, Categorical

ZERO = PointMass(0)

init = state = (0, Categorical(vals=[-48,-24, 24, 48]), Categorical(vals=[-8,-4, 4, 8]), Categorical(vals=[-8,-4, 4, 8]))
tree = [[1], [2, 3], [], []]

def node_value(node, state=None):
    """A distribution over total rewards after the given node."""
    # print("state: {}".format(state))
    # print("tree: {}".format(self.tree))
    # print([node_value(n1, state) for n1 in tree[node]])
    # print([state[n1] for n1 in tree[node]])
    # print([node_value(n1, state) + state[n1] for n1 in tree[node]])
    # print(list(expectation(self.node_value(n1, state) + state[n1]) for n1 in self.tree[node]))
    ret = max(
        (node_value(n1, state) + state[n1] for n1 in tree[node]),
        default=ZERO,
        key=expectation,
    )

    print("node: {}, state: {}".format(node, state[node]))
    print([state[n1] for n1 in tree[node]])
    print("Ret: {}".format(ret))

    return ret


state_1 = (0, 48, Categorical(vals=[-8,-4, 4, 8]), 4)
state_2 = (0, 48, 4, Categorical(vals=[-8,-4, 4, 8]))
state_3 = (0, 48, 4, Categorical(vals=[-8,-4, 4, 8]))


val = node_value(0, state_1)
print("\nFinal outcome:")
print(val)

print(Categorical(vals=[-24, 24]) + 4)

