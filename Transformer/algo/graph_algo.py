import torch
import numpy as np
import os

def find_path(path, i, j):
    if path[i, j] == 0:
        return [i, j]
    else:
        k = path[i, j]
        return find_path(path, i, k)[:-1] + find_path(path, k, j)

def floyd_warshall(adj):    
    if os.path.exists('floyd_warshall.npz'):
        mat = np.load("floyd_warshall.npz")
        dist,path=mat['a'],mat['b']
        return dist,path
    
    n = adj.shape[0]
    dist = adj.clone()
    path = torch.zeros_like(adj,dtype=torch.int64)
    dist[dist == 0] = float('inf')
    for i in range(n):
        dist[i][i] = 0
    count = 0
    for k in range(n):
        for i in range(n):
            for j in range(n):
                count+=1
                if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        path[i,j] = k
                print(f"{count}/{n**3}")
    dist[dist>=510]=510
    np.savez('floyd_warshall',a=dist.numpy(),b=path.numpy())
    return dist,path