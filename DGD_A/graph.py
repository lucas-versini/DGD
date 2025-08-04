from utils import build_W, plot_W, QuadraticDataset, LogisticDataset
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Agent parameters
m = 12 # Number of agents
n = 5 # Number of samples per agent
d = 2 # Dimension of the problem

# Function parameters
L = 1.
mu = 0.1

# Algorithm parameters
n_iter = 20000

# Gossip matrix
list_W = [a * build_W(m, type = "Four", strength = 0.9) + (1 - a) * build_W(m, type = "FL") for a in [0.2, 0.8]]
list_gamma = [1e-3, 5e-3]

dataset = LogisticDataset(m = m, n = n, d = d, gamma = list_gamma[0], W = list_W[0], n_iter = n_iter, L = L, mu = mu)
theta_star = dataset.theta_star.copy()  # shape (d)
Theta_star = np.tile(theta_star, (m, 1))  # shape (m, d)

n_runs = 5000

result = np.zeros((len(list_W), len(list_gamma), n_runs, m, d))

with open("configuration_small.pkl", "wb") as f:
    pickle.dump({
        "m": m,
        "n": n,
        "d": d,
        "L": L,
        "mu": mu,
        "n_iter": n_iter,
        "gamma": list_gamma,
        "list_W": list_W,
        "dataset": dataset,
        "list_W": list_W
    }, f)

for i, W in enumerate(list_W):
    print(f"Running for W index {i}/{len(list_W)}")
    dataset.set_W(W)
    for j, gamma in enumerate(list_gamma):
        print(f"Running for gamma index {j}/{len(list_gamma)}")
        dataset.set_gamma(gamma)
        for k in range(n_runs):
            dataset.run()
            result[i, j, k] = dataset.history[-1]

    with open("result_small.pkl", "wb") as f:
        pickle.dump(result, f)

