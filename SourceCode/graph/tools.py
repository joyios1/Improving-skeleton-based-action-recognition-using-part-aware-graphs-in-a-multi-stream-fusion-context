import numpy as np
import networkx as nx


# region graph functions
def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A
# endregion


# region Visualization
def create_ntu_node_positions():
    """
    returns a dictionary with the positions of each node
    """

    pos = {}
    # spine
    pos[0] = np.array([0.0, 0.0])
    pos[1] = np.array([0.0, 0.3])
    # chest
    pos[20] = np.array([0.0, 0.55])
    # neck
    pos[2] = np.array([0.0, 0.65])
    # head
    pos[3] = np.array([0.0, 0.8])
    # left arm
    pos[4] = np.array([0.1, 0.60])
    pos[5] = np.array([0.13, 0.25])
    pos[6] = np.array([0.15, 0])
    pos[7] = np.array([0.155, -0.1])
    pos[22] = np.array([0.16, -0.22])
    pos[21] = np.array([0.2, -0.28])
    # right arm
    pos[8] = np.array([-0.1, 0.60])
    pos[9] = np.array([-0.13, 0.25])
    pos[10] = np.array([-0.15, 0])
    pos[11] = np.array([-0.155, -0.1])
    pos[24] = np.array([-0.16, -0.22])
    pos[23] = np.array([-0.2, -0.28])
    # left leg
    pos[12] = np.array([0.075, -0.2])
    pos[13] = np.array([0.070, -0.5])
    pos[14] = np.array([0.065, -0.7])
    pos[15] = np.array([0.09, -0.88])
    # right leg
    pos[16] = np.array([-0.075, -0.2])
    pos[17] = np.array([-0.070, -0.5])
    pos[18] = np.array([-0.065, -0.7])
    pos[19] = np.array([-0.09, -0.88])
    return pos


def create_ntu_legs_positions():
    """
    returns a dictionary with the positions of each node
    """

    pos = {}
    # spine
    pos[0] = np.array([0.0, 0.0])
    # left leg
    pos[1] = np.array([0.075, -0.2])
    pos[2] = np.array([0.070, -0.5])
    pos[3] = np.array([0.065, -0.7])
    pos[4] = np.array([0.09, -0.88])
    # right leg
    pos[5] = np.array([-0.075, -0.2])
    pos[6] = np.array([-0.070, -0.5])
    pos[7] = np.array([-0.065, -0.7])
    pos[8] = np.array([-0.09, -0.88])
    return pos


def create_ntu_arms_positions():
    pos = {}
    # chest
    pos[8] = np.array([0.0, 0.55])
    # left arm
    pos[0] = np.array([0.1, 0.60])
    pos[1] = np.array([0.13, 0.25])
    pos[2] = np.array([0.15, 0])
    pos[3] = np.array([0.155, -0.1])
    pos[10] = np.array([0.16, -0.22])
    pos[9] = np.array([0.2, -0.28])
    # right arm
    pos[4] = np.array([-0.1, 0.60])
    pos[5] = np.array([-0.13, 0.25])
    pos[6] = np.array([-0.15, 0])
    pos[7] = np.array([-0.155, -0.1])
    pos[12] = np.array([-0.16, -0.22])
    pos[11] = np.array([-0.2, -0.28])

    return pos


def create_ntu_right_hand_left_leg_positions():
    pos = {}
    # spine
    pos[0] = np.array([0.0, 0.0])
    pos[1] = np.array([0.0, 0.3])
    # chest
    pos[10] = np.array([0.0, 0.55])
    # right arm
    pos[2] = np.array([-0.1, 0.60])
    pos[3] = np.array([-0.13, 0.25])
    pos[4] = np.array([-0.15, 0])
    pos[5] = np.array([-0.155, -0.1])
    pos[12] = np.array([-0.16, -0.22])
    pos[11] = np.array([-0.2, -0.28])
    # left leg
    pos[6] = np.array([0.075, -0.2])
    pos[7] = np.array([0.070, -0.5])
    pos[8] = np.array([0.065, -0.7])
    pos[9] = np.array([0.09, -0.88])
    return pos


def create_ntu_left_hand_right_leg_positions():
    pos = {}
    # spine
    pos[0] = np.array([0.0, 0.0])
    pos[1] = np.array([0.0, 0.3])
    # chest
    pos[10] = np.array([0.0, 0.55])
    # left arm
    pos[2] = np.array([0.1, 0.60])
    pos[3] = np.array([0.13, 0.25])
    pos[4] = np.array([0.15, 0])
    pos[5] = np.array([0.155, -0.1])
    pos[12] = np.array([0.16, -0.22])
    pos[11] = np.array([0.2, -0.28])
    # right leg
    pos[6] = np.array([-0.075, -0.2])
    pos[7] = np.array([-0.070, -0.5])
    pos[8] = np.array([-0.065, -0.7])
    pos[9] = np.array([-0.09, -0.88])
    return pos


def draw_part_aware_graph(g, ax, mode='full_graph', node_size=100, node_color='red', edge_color='grey'):

    if mode == 'full_graph':
        pos = create_ntu_node_positions()
    elif mode == 'hands':
        pos = create_ntu_arms_positions()
    elif mode == 'rhll':
        pos = create_ntu_right_hand_left_leg_positions()
    elif mode == 'lhrl':
        pos = create_ntu_left_hand_right_leg_positions()
    else:
        pos = create_ntu_legs_positions()

    nx.draw_networkx(g, ax=ax, pos=pos, node_size=node_size, node_color=node_color, edge_color=edge_color,
                     arrows=True, with_labels=False)
# endregion
