import matplotlib.pyplot as plt
import networkx as nx
import pytest

from mouselab.graph_utils import adjacency_list_from_graph, graph_from_adjacency_list

PLOT = False

graph_utils_test_cases = [
    [
        "symmetric",
        [
            [1, 5, 9],
            [2],
            [3, 4],
            [],
            [],
            [6],
            [7, 8],
            [],
            [],
            [10],
            [11, 12],
            [],
            [],
        ],
    ],
    ["nonsymmetric", [[1, 5, 7], [2], [3, 4], [], [], [6], [], [8], [9], []]],
]


@pytest.mark.parametrize("name,adjacency_list", graph_utils_test_cases)
def test_adjacency_functions(name, adjacency_list):
    env = graph_from_adjacency_list(adjacency_list)

    if PLOT:  # global boolean for plotting
        nx.draw(env, pos=nx.planar_layout(env))
        plt.show()

    # tests if dictionary keys are same as all possible (state,action) pairs
    assert adjacency_list_from_graph(env) == adjacency_list
