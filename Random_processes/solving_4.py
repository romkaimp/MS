import numpy as np
import numpy.linalg as la
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


np.set_printoptions(precision=2)
s = int(input())
matrix = np.zeros(shape=(s, s))

k = []
X = []
for i in range(s):
    inp = list(map(int, input().split()))
    k.append(inp[0])
    if k[i] == 0:
        X.append(i)
        matrix[i, i] = 1
    for j in range(1, len(inp)):
        matrix[i][inp[j] - 1] = 1/k[i]
weights = list(map(float, input().split()))
weights = np.array(weights) / np.sum(weights)

init_mat = matrix.copy()

initial_st = np.zeros(s)
initial_st[0] = 1

initial_st = torch.tensor(initial_st, requires_grad=False)
matrix = torch.tensor(matrix, requires_grad=True)
in_mat = torch.tensor(init_mat, requires_grad=False)
weights = torch.tensor(weights, requires_grad=False)

optimizer = optim.Adam([matrix], lr=1)
criterion = nn.MSELoss()

res = torch.zeros(s)

for i in range(400):
    stationary_matrix = initial_st @ matrix

    for i in range(s):
        stationary_matrix = stationary_matrix @ matrix

    optimizer.zero_grad()

    loss = 5 * criterion(stationary_matrix, weights)
    loss += 5 * criterion(torch.sum(matrix, dim=-1), torch.ones(s, dtype=torch.float64))
    loss += 2 * criterion(matrix * torch.eye(s , dtype=torch.float64), torch.zeros(s, dtype=torch.float64))


    loss.backward()
    optimizer.step()

matrix = matrix.detach().numpy()

for i in range(s):
    for j in range(s):
        if init_mat[i, j] != 0 and i != j:
            print(i + 1, j + 1, matrix[i, j])

"""
5
4 3 4 5 2
0
2 5 4
0
0
0 3 0 4 5
"""