import sys
import numpy as np
import networkx as nx
from SourceCode.graph import tools
import matplotlib.pyplot as plt

sys.path.extend(['../'])

full_graph = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
              (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
              (19, 18), (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)]

arms = [(5, 9), (1, 9), (12, 13), (13, 8), (8, 7), (7, 6), (6, 5), (10, 11), (11, 4), (4, 3), (3, 2), (2, 1)]

legs = [(9, 8), (8, 7), (7, 6), (6, 1), (5, 4), (4, 3), (3, 2), (2, 1)]

rhll = [(13, 6), (12, 6), (10, 9), (9, 8), (8, 7), (7, 1), (6, 5), (5, 4), (4, 3), (3, 11), (2, 11), (1, 2)]

lhrl = [(13, 6), (12, 6), (10, 9), (9, 8), (8, 7), (7, 1), (6, 5), (5, 4), (4, 3), (3, 11), (2, 11), (1, 2)]


class Graph:
    def __init__(self, part_aware_graph):

        if part_aware_graph == 'full_graph':
            self.get_final_adjacency(full_graph, 25)
        elif part_aware_graph == 'arms':
            self.get_final_adjacency(arms, 13)
        elif part_aware_graph == 'legs':
            self.get_final_adjacency(legs, 9)
        elif part_aware_graph == 'rhll':
            self.get_final_adjacency(rhll, 13)
        elif part_aware_graph == 'lhrl':
            self.get_final_adjacency(lhrl, 13)
        else:
            raise ValueError('Part aware graph is not supported!!!')

        # fig, ax = plt.subplots()
        # g = nx.from_numpy_array(self.A[1], create_using=nx.DiGraph)
        # tools.draw_part_aware_graph(g, ax, mode=part_aware_graph)
        # plt.show()

    def get_final_adjacency(self, part_aware_graph, number_of_nodes):
        self_loops = [(i, i) for i in range(number_of_nodes)]
        outwards = [(i - 1, j - 1) for (i, j) in part_aware_graph]
        inwards = [(j, i) for (i, j) in outwards]
        self.A = tools.get_spatial_graph(number_of_nodes, self_loops, inwards, outwards)

