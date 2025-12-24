
from typing import Tuple
import torch

def _sensitive_from_degree(edge_index, num_nodes):
    row, col = edge_index
    deg = torch.bincount(row, minlength=num_nodes)
    thresh = torch.median(deg.float())
    return (deg >= thresh).long()

def _edge_index_to_nx(edge_index, num_nodes):
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    ei = edge_index.cpu().numpy()
    for u, v in zip(ei[0], ei[1]):
        if int(u) != int(v):
            g.add_edge(int(u), int(v))
    return g

def _sensitive_from_eigen(edge_index, num_nodes):
    import torch
    g = _edge_index_to_nx(edge_index, num_nodes)
    import networkx as nx
    cen = nx.eigenvector_centrality_numpy(g)
    vals = torch.tensor([cen[i] for i in range(num_nodes)], dtype=torch.float32)
    thresh = torch.median(vals)
    return (vals >= thresh).long()

def _sensitive_from_betw(edge_index, num_nodes):
    import torch
    g = _edge_index_to_nx(edge_index, num_nodes)
    import networkx as nx
    b = nx.betweenness_centrality(g, normalized=True)
    vals = torch.tensor([b[i] for i in range(num_nodes)], dtype=torch.float32)
    thresh = torch.median(vals)
    return (vals >= thresh).long()

def load_planetoid(name: str = 'Cora') -> Tuple:
    import torch_geometric.transforms as T
    from torch_geometric.datasets import Planetoid
    name = name.capitalize()
    dataset = Planetoid(root=f'data/{name}', name=name, transform=T.NormalizeFeatures())
    data = dataset[0]
    data.S = _sensitive_from_degree(data.edge_index, data.num_nodes)
    return dataset, data

def load_local_sbm(n_per_block: int = 800, ratio: float = 0.5, p_in: float = 0.08, p_out: float = 0.01,
                   feat_dim: int = 128, seed: int = 42, sensitive: str = 'degree'):
    """Offline 2-block SBM with tunable imbalance & homophily.
       ratio: fraction in block A (0.7 => 70/30). sensitive: degree|eigen|betw
    """
    import numpy as np
    import networkx as nx
    from torch_geometric.utils import to_undirected
    from torch_geometric.data import Data

    total = n_per_block * 2
    n_a = max(int(total * ratio), 2)
    n_b = max(total - n_a, 2)

    g = nx.stochastic_block_model([n_a, n_b],
                                  [[p_in, p_out], [p_out, p_in]], seed=seed)

    n = g.number_of_nodes()
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1.0, size=(n, feat_dim)).astype('float32')
    y = np.zeros(n, dtype='int64')
    y[n_a:] = 1
    X[:n_a] += 0.25; X[n_a:] -= 0.25

    edges = np.array(list(g.edges), dtype=np.int64).T
    if edges.size == 0:
        edges = np.zeros((2, 0), dtype=np.int64)
    edge_index = torch.from_numpy(edges)
    edge_index = to_undirected(edge_index, num_nodes=n)

    data = Data(x=torch.from_numpy(X), y=torch.from_numpy(y), edge_index=edge_index)
    data.x = (data.x - data.x.mean(0, keepdim=True)) / (data.x.std(0, keepdim=True) + 1e-6)

    torch.manual_seed(seed)
    n_train = int(0.6 * n); n_val = int(0.2 * n)
    perm = torch.randperm(n)
    train_idx = perm[:n_train]; val_idx = perm[n_train:n_train+n_val]; test_idx = perm[n_train+n_val:]
    data.train_mask = torch.zeros(n, dtype=torch.bool); data.train_mask[train_idx] = True
    data.val_mask = torch.zeros(n, dtype=torch.bool); data.val_mask[val_idx] = True
    data.test_mask = torch.zeros(n, dtype=torch.bool); data.test_mask[test_idx] = True

    sensitive = (sensitive or 'degree').lower()
    if sensitive in ['degree','deg']:
        data.S = _sensitive_from_degree(data.edge_index, n)
    elif sensitive in ['eigen','eig','eigenvector']:
        data.S = _sensitive_from_eigen(data.edge_index, n)
    elif sensitive in ['betweenness','betw']:
        data.S = _sensitive_from_betw(data.edge_index, n)
    else:
        raise ValueError(f"Unknown sensitive='{sensitive}'")
    class DSLike:
        def __init__(self, num_features, num_classes): 
            self.num_features = num_features; self.num_classes = num_classes
    ds_like = DSLike(num_features=data.x.size(1), num_classes=2)
    return ds_like, data

def load_dataset(name: str = 'Cora', **kwargs):
    name_lower = name.lower()
    if name_lower in ['cora', 'citeseer', 'pubmed']:
        return load_planetoid(name)
    elif name_lower in ['localsbm','sbm','local']:
        return load_local_sbm(**kwargs)
    else:
        raise ValueError(f'Unknown dataset: {name}')
