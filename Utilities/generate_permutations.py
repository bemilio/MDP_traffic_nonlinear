import torch

# generate vectors of length N that sum to k. Warning: complexity is exponential
def generate_permutations(k, N):
    if N == 1:
        x = torch.tensor([[k]])
    else:
        if k == 0:
            x = torch.zeros(1, N)
        else:
            x = torch.zeros(0, N)
            for i in range(k+1):
                y = generate_permutations(k - i, N - 1)
                y = torch.cat((y, i * torch.ones(y.size(0), 1)), 1)
                x = torch.cat((x, y), 0)
    return x