import numpy
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
import random
from graphviz import Digraph

def plot_graph(matrix, label="markov_chain", states=None):
    dot = Digraph(format='png', engine='dot')
    num_states = len(matrix)

    for i in range(num_states):
        if states is None:
            dot.node(str(i), f"State {i + 1}")
        else:
            dot.node(str(i), states[i])

    for i in range(num_states):
        for j in range(num_states):
            prob = matrix[i][j]
            if prob > 0:
                dot.edge(str(i), str(j), label=f"{prob:.2f}")

    dot.render(label, view=True)

def find_stationary_matrices(matrix: numpy.ndarray, num_alternatives = 1, verbose = False):
    num_states = matrix.shape[2]
    if matrix.ndim == 3:
        num_alternatives = matrix.shape[2]

    stationary_matrices = np.empty((num_alternatives, num_states, num_states))

    for h in range(num_alternatives):
        SLAU = np.empty((num_states, num_states))
        b = np.empty(num_states)
        for i in range(num_states - 1):
            for j in range(num_states):
                SLAU[i, j] = matrix[h, j, i]
            b[i] = 0
            SLAU[i, i] -= 1

        # normalizing
        for i in range(num_states):
            SLAU[-1, i] = 1
        b[-1] = 1

        if verbose:
            print("SLAU:", SLAU)
            print("b:", b)

        stationary_probs = la.solve(SLAU, b)

        stationary_matrices[h] = stationary_probs

    if num_alternatives == 1:
        return stationary_matrices[0]
    return stationary_matrices