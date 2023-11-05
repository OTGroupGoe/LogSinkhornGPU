
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from LogSinkhornGPU import LogSinkhornTorch

def batch_iteration(pi, x, y, muref, nuref, eps, partition, device='cuda'):
    # Get maximum size of cell marginals
    B = len(partition)
    max_size = np.max(list(map(len, partition)))
    Mu = torch.zeros((B, max_size), device=device)
    Nu = torch.zeros((B, len(y)), device=device)
    Muref = torch.zeros((B, max_size), device=device)
    Nuref = torch.zeros((B, len(y)), device=device)
    C = torch.zeros((B, max_size, len(y)), device=device)
    # Batch problems
    for (i, J) in enumerate(partition):
        piJ = pi[J, :]
        xJ = x[J]
        muJ = torch.sum(piJ, dim=1).ravel()
        nuJ = torch.sum(piJ, dim=0).ravel()
        Mu[i, :len(J)] = muJ
        Nu[i, :] = nuJ
        # This might be different in the unbalanced case:
        Muref[i, :len(J)] = muref[0,J] 
        Nuref[i, :] = nuref[0,:]
        C[i, :len(J), :] = (xJ.reshape(-1,1) - y.reshape(1,-1))**2
    solver = LogSinkhornTorch(Mu, Nu, C, eps, muref=Muref, nuref=Nuref)

    solver.iterate_until_max_error()
    # Retrieve solution
    pi_batch = solver.get_pi_dense()
    pi_new = torch.zeros_like(pi)
    for (i, J) in enumerate(partition):
        pi_new[J,:] = pi_batch[i, :len(J), :]
    return pi_new

def main():
    device = 'cuda'
    M = N = 32
    muref = torch.ones(1, M, device=device)
    nuref = torch.ones(1, N, device=device)
    muref = muref/torch.sum(muref, dim=1, keepdim=True)
    nuref = nuref/torch.sum(nuref, dim=1, keepdim=True)
    x = torch.linspace(0, 1, M, device=device)
    y = torch.linspace(0, 1, N, device=device)

    # PARTITIONS
    partA = [[i, i+1] for i in range(0, N, 2)]
    partB = [[0], *[[i, i+1] for i in range(1, N-2, 2)], [N-1]]

    print(f'A has {len(partA)} cells:\n', partA)
    print(f'B has {len(partB)} cells:\n', partB)

    # INITIAL PLAN
    # pi = torch.ones((M, N), device=device) / (M*N) # prod initialization
    pi = torch.fliplr(torch.eye(M, device=device))/M

    # PARAMETERS
    eps = 1/M**2
    n_iter = 25

    cols = 5
    rows = (n_iter-1)//cols + 1
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(2*cols, 2*rows))

    for i, ax in tqdm(enumerate(axs.flatten())):
        partition = partA if i%2 == 0 else partB
        pi = batch_iteration(pi, x, y, muref, nuref, eps, partition)
        ax.imshow(pi.cpu().detach().T, origin="lower")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
