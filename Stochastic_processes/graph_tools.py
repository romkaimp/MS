import numpy as np
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