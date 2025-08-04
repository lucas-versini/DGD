import numpy as np
import scipy
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

seed = 0
np.random.seed(seed)

"""
This script contains:
- Functions to create communication matrices (stochastic, symmetric)
- Classes to run the DSGD-A algorithm on both quadratic and logistic functions
"""

### Communication matrix ###

def normalize_row(W):
  return W / np.sum(W, axis = 1)[:, None]
def normalize_col(W):
  return W / np.sum(W, axis = 0)[None, :]
def turn_doubly_stochastic(W):
  """ Turn a matrix W into a doubly stochastic matrix by iteratively normalizing rows and columns."""
  while not (np.allclose(W.sum(axis = 0), 1, 1e-13) and np.allclose(W.sum(axis = 1), 1, 1e-13)):
    W = normalize_col(normalize_row(W))
  return W

def build_W(m, type = "RandomTight", strength = .1):
  """ Builds a communication matrix.
   Args:
    m: number of agents
    type: type of communication matrix, can be "FL", "RandomConnected", "RandomTight", "Four"
    strength: strength of the connection between groups in the "Four" type

   Returns:
    W: a doubly stochastic matrix of shape (m, m)
  """
  if type == "FL":
    return np.ones((m, m)) * 1/m
  elif type == "RandomConnected":
    W = turn_doubly_stochastic(np.random.rand(m, m))
  elif type == "Circle":
    W = np.zeros((m, m))
    for i in range(m):
      W[i, (i + 1) % m] = 1 / 2
      W[i, (i - 1) % m] = 1 / 2
    W = turn_doubly_stochastic(W)
  elif type == "RandomTight":
    W = np.random.rand(m, m)
    i, j = m // 2 - 1, m // 2

    for a in range(i + 1):
      for b in range(j, m):
        if a != i or b != j:
          W[a, b] = W[b, a] = 0

    W = turn_doubly_stochastic(W)
  elif type == "Four":
    # 4 groups, each group is fully connected, and groups are connected to each other based on strength
    W = np.zeros((m, m))
    group_size = m // 4
    for i in range(4):
      start = i * group_size
      end = (i + 1) * group_size if i < 3 else m
      W[start:end, start:end] = np.ones((end - start, end - start)) * (1 / (end - start))
    for i in range(4):
      for j in range(i + 1, 4):
        start_i = i * group_size
        end_i = (i + 1) * group_size if i < 3 else m
        start_j = j * group_size
        end_j = (j + 1) * group_size if j < 3 else m
        idx_i = start_i # np.random.randint(start_i, end_i)
        idx_j = start_j # np.random.randint(start_j, end_j)
        W[idx_i, idx_j] = W[idx_j, idx_i] = strength
    W = turn_doubly_stochastic(W)

  W = (W + W.T) / 2  # Make it symmetric
  min_eigenvalue = np.min(np.linalg.eigvalsh(W))
  if min_eigenvalue < 1e-10:
    coef = -min_eigenvalue / (1 - min_eigenvalue) + np.random.rand() * 1e-1
    W = (1 - coef) * W + coef * np.eye(m)

  return W
def plot_W(W, ax = None, structure = None, title = None):
    """ Plots the communication graph represented by the matrix W."""
    m = W.shape[0]
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_title(title if title is not None else "Communication graph")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    if structure == "RandomTight" or structure is None:
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                if W[i, j] > 0:
                    ax.plot([np.cos(i * 2 * np.pi / m), np.cos(j * 2 * np.pi / m)],
                            [np.sin(i * 2 * np.pi / m), np.sin(j * 2 * np.pi / m)],
                            linewidth=W[i, j] * 6, color='black')
        if structure == "RandomTight":
          for i in range(W.shape[0]):
              ax.plot(np.cos(i * 2 * np.pi / m), np.sin(i * 2 * np.pi / m), 'o', color='red' if i <= m // 2 - 1 else "blue")
        else:
          for i in range(W.shape[0]):
              ax.plot(np.cos(i * 2 * np.pi / m), np.sin(i * 2 * np.pi / m), 'o', color='blue')
    elif structure == "Four":
        size_group = W.shape[0] // 4
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                if W[i, j] > 0:
                    group_i = min(i // size_group, 3)
                    group_j = min(j // size_group, 3)
                    center_i = 0.5 * np.array([np.cos(group_i * 2 * np.pi / 4), np.sin(group_i * 2 * np.pi / 4)])
                    center_j = 0.5 * np.array([np.cos(group_j * 2 * np.pi / 4), np.sin(group_j * 2 * np.pi / 4)])
                    size_i = size_group if group_i < 3 else W.shape[0] - 3 * size_group
                    size_j = size_group if group_j < 3 else W.shape[0] - 3 * size_group
                    ax.plot([center_i[0] + np.cos(i % size_i * 2 * np.pi / size_i) * 0.3,
                             center_j[0] + np.cos(j % size_j * 2 * np.pi / size_j) * 0.3],
                            [center_i[1] + np.sin(i % size_i * 2 * np.pi / size_i) * 0.3,
                             center_j[1] + np.sin(j % size_j * 2 * np.pi / size_j) * 0.3],
                            linewidth=W[i, j] * 10, color='black')  
        for i in range(4):
            center = 0.5 * np.array([np.cos(i * 2 * np.pi / 4), np.sin(i * 2 * np.pi / 4)])
            size = size_group if i < 3 else W.shape[0] - 3 * size_group
            for j in range(size):
                ax.plot(center[0] + np.cos(j * 2 * np.pi / size) * 0.3,
                        center[1] + np.sin(j * 2 * np.pi / size) * 0.3, 'o', color= ["red", "blue", "green", "orange"][i])
    # no axis
    ax.axis('off')
    # plt.show()

### Datasets ###
class Dataset:
    """ Meta-class for datasets used in the experiments."""

    def __init__(self, m, n, d, gamma, W, n_iter = 1000, L = 1., mu = 0.1):
        self.m = m
        self.n = n
        self.d = d
        self.gamma = gamma
        self.W = W
        self.n_iter = n_iter
        self.L = L
        self.mu = mu

        self.colors = np.random.rand(m, 3)

        self.get_dataset()
        self.get_lim_star()

    def get_dataset(self):
        """ Creates the dataset (matrices, vectors, etc.) """
        raise NotImplementedError("This method should be implemented by subclasses")

    def get_lim_star(self):
        """ Computes the limiting point and the optimum """
        raise NotImplementedError("This method should be implemented by subclasses")

    def noisy_grad(self, Theta):
        """ Returns a noisy gradient """
        raise NotImplementedError("This method should be implemented by subclasses")

    def run(self):
        """ Runs the D(S)GD-A algorithm """
        if not hasattr(self, 'history'):
            Theta = np.random.randn(self.m, self.d)
        else:
            Theta = self.history[0]
        history = np.zeros((self.n_iter + 1, self.m, self.d))
        history[0] = Theta.copy()

        for i in range(self.n_iter):
            # Compute a noisy gradient
            grad = self.noisy_grad(Theta)

            # Update the parameters
            Theta = self.W @ (Theta - self.gamma * grad)

            history[i + 1] = Theta.copy()
        
        self.history = history.copy()

    def plot_distance(self, idx_end = -1, legend = True, save = None):
        """ Plots the evolution of the distance to the limiting point and to the optimum """
        history_dist_star = ((self.history - self.theta_star[np.newaxis, :])**2).sum((1, 2))
        history_dist_lim = ((self.history - self.Theta_lim.reshape(self.m, self.d))**2).sum((1, 2))
        dist_star_lim = ((self.Theta_lim.reshape(self.m, self.d) - self.theta_star[np.newaxis, :])**2).sum()

        plt.figure(figsize = (8, 5))
        idx_start, idx_end = 10, idx_end
        plt.plot(history_dist_lim[idx_start:idx_end], label=r"$\|\Theta_t - \Theta_{\text{lim}}\|^2$")
        plt.plot(history_dist_star[idx_start:idx_end], label=r"$\|\Theta_t - \Theta^*\|^2$")
        plt.axhline(dist_star_lim, color='red', linestyle='--', label=r"$\|\Theta_{\text{lim}} - \Theta^*\|^2$")
        plt.title("Distance to limit point and local optimum")
        plt.xlabel("Iteration t")
        plt.ylabel("Distance")
        plt.grid()
        if legend:
          plt.legend()
        if save is not None:
            plt.savefig(save + ".png", dpi=200)
            plt.savefig(save + ".pdf", dpi=200)

        plt.show()
        
    def plot_2D(self, ax = None, title = None, interval = 1, legend = True, save = None, delta = .5, end = -1, local = True, x_min = None, x_max = None, y_min = None, y_max = None):
      """ Plots the trajectories in 2D """
      if self.d != 2:
          raise ValueError("This method can only be used for 2D datasets")

      if ax is None:
          fig, ax = plt.subplots(figsize=(8, 5))

      Theta_lim = self.Theta_lim.reshape(self.m, self.d)
      
      # Clear the axis first if you want to redraw
      ax.cla()

      for i in range(self.m):
          ax.scatter(self.history[0, i, 0], self.history[0, i, 1], color=self.colors[i], s=50, linewidth=1,
                    label='Agent start' if i == 0 else "", alpha=0.4, marker='x')
          ax.plot(self.history[:end][::interval, i, 0], self.history[:end][::interval, i, 1], color=self.colors[i], marker='o', markersize=1,
                  linestyle='-', alpha=0.4)
      for i in range(self.m):
          ax.scatter(Theta_lim[i, 0], Theta_lim[i, 1], zorder=5, color=self.colors[i], edgecolors="black", s=25,
                    linewidth=1, label=r'Agent limit point $\theta^{\text{lim}}_k$' if i == 0 else "")
          theta_star_loc = self.Theta_star_loc[i].mean(0)
          if local:
            ax.scatter(theta_star_loc[0], theta_star_loc[1], color=self.colors[i], zorder=5, s=50, marker='*',
                      label=r'Agent local optimum $\theta^{*}_k$' if i == 0 else "")

      ax.scatter(self.theta_star[0], self.theta_star[1], color='blue', label=r'Global optimum $\theta^{*}$', zorder=6, s=200,
                edgecolors="black", linewidth=1, marker='^', alpha=0.2)

      ax.set_title(title if title is not None else 'Trajectory of agents')
      if legend:
        ax.legend()
      ax.grid()

      if x_min is None or x_max is None or y_min is None or y_max is None:
        x_min, x_max = Theta_lim[:, 0].min() - delta, Theta_lim[:, 0].max() + delta
        y_min, y_max = Theta_lim[:, 1].min() - delta, Theta_lim[:, 1].max() + delta
      ax.set_xlim(x_min, x_max)
      ax.set_ylim(y_min, y_max)

      if save is not None:
        plt.savefig(save + ".png", dpi=200)
        plt.savefig(save + ".pdf", dpi=200)
    
    def plot_legend_only(self, path):
      fig, ax = plt.subplots(figsize=(2.8, 1.5))

      # Plot invisible elements just to include in the legend
      ax.plot([], [], color='gray', marker='x', linestyle='None', label='Agent start')
      ax.plot([], [], color='gray', linestyle='-', marker='o', markersize=3, label='Agent trajectory')
      ax.scatter([], [], color='gray', edgecolors="black", s=25, label=r'Agent limit point $\theta^{\text{lim}}_k$')
      ax.scatter([], [], color='gray', s=50, marker='*', label=r'Agent local optimum $\theta^{*}_k$')
      ax.scatter([], [], color='blue', s=200, edgecolors="black", marker='^', alpha=0.2, label=r'Global optimum $\theta^{*}$')

      # Display only the legend
      legend = ax.legend(loc='center', frameon=False)
      ax.axis('off')

      plt.tight_layout()
      plt.savefig(path + ".pdf", dpi=200)
    
    def plot_distance_legend_only(self, path):
      plt.figure(figsize=(1.5, 1))

      # Dummy lines to generate legend
      line1 = Line2D([0], [0], label=r"$\|\Theta_t - \Theta_{\text{lim}}\|^2$", color='C0')
      line2 = Line2D([0], [0], label=r"$\|\Theta_t - \Theta^*\|^2$", color='C1')
      line3 = Line2D([0], [0], label=r"$\|\Theta_{\text{lim}} - \Theta^*\|^2$", color='red', linestyle='--')

      plt.legend(handles=[line1, line2, line3], loc='center', frameon=False, ncol=1)
      plt.axis('off')
      plt.tight_layout()
      plt.savefig(path + ".pdf", dpi=200)
      plt.show()
    
    def set_gamma(self, gamma):
        """ Changes the value of the step size gamma """
        self.gamma = gamma
        self.get_lim_star()
    
    def set_W(self, W):
        """ Changes the value of the communication matrix W """
        self.W = W
        self.get_lim_star()

class QuadraticDataset(Dataset):
    """ Quadratic dataset with a local optimum and a limit point."""

    def get_dataset(self):
        Theta_star_loc = np.random.randn(self.m, self.n, self.d)
        offset = np.random.randn(self.m, self.d) * 0.2
        Theta_star_loc += offset[:, None, :]
        A = np.random.randn(self.m, self.n, self.d, self.d) * 1.
        A = np.einsum('ijkl,ijnl->ijkn', A, A)
        eigenvalues = np.linalg.eigvalsh(A)
        eig_min, eig_max = np.min(eigenvalues), np.max(eigenvalues)
        A = (A - eig_min * np.eye(self.d)[None, None, :, :]) / (eig_max - eig_min + 1e-10) * (self.L - self.mu) + self.mu * np.eye(self.d)[None, None, :, :] * np.ones((self.m, self.n, 1, 1))
        
        self.Theta_star_loc, self.A = Theta_star_loc, A
    
    def get_lim_star(self):
        # Theta_lim
        A_equiv = self.A.mean(1)
        mean_equiv = np.einsum('mnij,mnj->mi', self.A, self.Theta_star_loc) / self.n
        mean_equiv = np.einsum('mij,mj->mi', np.linalg.inv(A_equiv), mean_equiv)

        A_bar_tens = np.kron(np.eye(self.m), A_equiv.mean(0))
        A_tens = scipy.linalg.block_diag(*A_equiv)

        P = np.kron(1/self.m * np.ones((self.m, self.m)), np.eye(self.d))
        I_tens = np.eye(self.m * self.d)
        W_tens = np.kron(self.W, np.eye(self.d))
        B = np.linalg.pinv(I_tens - W_tens) @ W_tens
        C = np.linalg.pinv(I_tens + self.gamma * B @ A_tens) @ B @ A_tens
        H = P @ np.linalg.inv(A_bar_tens) @ A_tens
        
        # Exact
        cons_lim = H @ mean_equiv.flatten() - self.gamma * H @ np.linalg.inv(I_tens - self.gamma * C @ H) @ C @ (I_tens - H) @ mean_equiv.flatten()
        disag_lim = - self.gamma * np.linalg.inv(I_tens - self.gamma * C @ H) @ C @ (H - I_tens) @ mean_equiv.flatten()
        
        # Taylor expansion
        # cons_lim = H @ mean_equiv.flatten() - self.gamma * H @ B @ A_tens @ (I_tens - H) @ mean_equiv.flatten() + self.gamma**2 * H @ B @ A_tens @ (I_tens - H) @ B @ A_tens @ (I_tens - H) @ mean_equiv.flatten()
        # disag_lim = self.gamma * B @ A_tens @ (I_tens - H) @ mean_equiv.flatten() - self.gamma**2 * B @ A_tens @ (I_tens - H) @ B @ A_tens @ (I_tens - H) @ mean_equiv.flatten()

        Theta_lim = cons_lim + disag_lim

        theta_star = np.linalg.solve(A_equiv.sum(0), np.einsum('mij,mj->i', A_equiv, mean_equiv))

        self.Theta_lim, self.theta_star = Theta_lim, theta_star

    def noisy_grad(self, Theta):
        # Randomly sample a sample for each agent
        idx = np.random.randint(0, self.n, size = self.m)
        A_ = self.A[np.arange(self.m), idx]
        Theta_star_loc_ = self.Theta_star_loc[np.arange(self.m), idx]

        return np.einsum('mij,mj->mi', A_, Theta - Theta_star_loc_)

class LogisticDataset(Dataset):
    """ Logistic dataset with a local optimum and a limit point."""

    def get_theta_star(self, n_iter = 50000, tol = 0.):
        theta = np.random.randn(self.d)
        # step = 1 / (2 * self.L)
        step = 5e-3

        for _ in range(n_iter):
            temp_grad = 1. / (1 + np.exp(-self.Theta_star_loc @ theta))
            grad = np.einsum('mn,mnd->md', temp_grad, self.Theta_star_loc).mean(axis=0) / self.n + self.mu * theta
            theta -= step * grad

            if np.linalg.norm(grad) < tol:
                break

        self.theta_star = theta
    def get_dataset(self):

        Theta_star_loc = np.random.randn(self.m, self.n, self.d)
        offset = np.random.randn(self.m, self.d) * 2
        Theta_star_loc += offset[:, None, :]

        L_array = np.max(1/4 * (Theta_star_loc ** 2).sum(-1).mean(1))
        Theta_star_loc *= np.sqrt((self.L - self.mu) / L_array)
        self.Theta_star_loc = Theta_star_loc

        self.get_theta_star()

        coeff = np.exp(-Theta_star_loc @ self.theta_star)
        coeff = coeff / (1 + coeff) ** 2 # shape (m, n)

        local_hessian = np.einsum('mni,mnj->mnij', self.Theta_star_loc, self.Theta_star_loc)

        A = np.einsum('mnij,mn->mnij', local_hessian, coeff).mean(1) # Shape: (m, d, d)
        A = A + self.mu * np.eye(self.d)[None, :, :] * np.ones((self.m, 1, 1))

        self.A = A

    def get_lim_star(self):
      A_bar = self.A.mean(0)

      Theta_star = np.tile(self.theta_star, self.m).reshape(self.m, self.d)
      grad_star = (np.einsum('mnd,mn->md', self.Theta_star_loc, 1 / (1 + np.exp(-np.einsum('mnd,md->mn', self.Theta_star_loc, Theta_star)))) / self.n + self.mu * Theta_star)

      eigenvalues, eigenvectors = np.linalg.eigh(self.W)
      idx = np.argsort(eigenvalues)[::-1]
      eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
      assert np.allclose(self.W, eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T)
      diag = 1 - eigenvalues
      diag[0] = 0.
      diag_inv = np.linalg.pinv(np.diag(diag))
      B = eigenvectors @ (diag_inv @ np.diag(eigenvalues)) @ eigenvectors.T

      disag_lim = - self.gamma * B @ grad_star
      cons_lim = self.theta_star - np.einsum('mij,mj->mi', np.einsum('ij,mjk->mik', np.linalg.inv(A_bar), self.A), disag_lim) / self.m

      Theta_lim = (cons_lim + disag_lim).flatten()

      self.Theta_lim = Theta_lim

    def noisy_grad(self, Theta):
        idx = np.random.randint(0, self.n, size = self.m)
        Theta_star_loc_ = self.Theta_star_loc[np.arange(self.m), idx]

        return np.einsum('md,m->md', Theta_star_loc_, 1 / (1 + np.exp(-np.einsum('md,md->m', Theta_star_loc_, Theta)))) + self.mu * Theta