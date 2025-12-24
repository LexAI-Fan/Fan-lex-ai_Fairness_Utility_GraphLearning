
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling, to_undirected
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from utils import set_seed, save_results
from data import load_dataset
from models import LinkPredictor
from fairness import demographic_parity_gap, corr_with_sensitive

def split_edges(edge_index, val_ratio=0.05, test_ratio=0.1):
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges)
    test_size = int(test_ratio * num_edges)
    val_size = int(val_ratio * num_edges)
    test_pos = edge_index[:, perm[:test_size]]
    val_pos = edge_index[:, perm[test_size:test_size+val_size]]
    train_pos = edge_index[:, perm[test_size+val_size:]]
    return to_undirected(train_pos), to_undirected(val_pos), to_undirected(test_pos)

def sample_negatives(edge_index, num_nodes, num_samples):
    return negative_sampling(edge_index=edge_index, num_nodes=num_nodes, num_neg_samples=num_samples, method='sparse')

def edge_sensitive_attr(S, edge_index):
    src, dst = edge_index
    return torch.maximum(S[src], S[dst])

def train_epoch(model, x, full_edge_index, pos_edges, S_node, optimizer,
                fair_lambda=0.0, num_neg=None):
    model.train()
    optimizer.zero_grad()

    if num_neg is None:
        num_neg = 4 * pos_edges.size(1)
    neg_edges = sample_negatives(full_edge_index, x.size(0), num_neg)

    pos_score, neg_score, _ = model(x, full_edge_index, pos_edges, neg_edges)

    # BCE on pos+neg
    bce = F.binary_cross_entropy_with_logits(
        torch.cat([pos_score,               neg_score]),
        torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
    )

    # DP penalty on pos+neg edges
    S_pos = edge_sensitive_attr(S_node, pos_edges)
    S_neg = edge_sensitive_attr(S_node, neg_edges)
    pos_prob = torch.sigmoid(pos_score)
    neg_prob = torch.sigmoid(neg_score)

    prob_all = torch.cat([pos_prob, neg_prob])
    S_all    = torch.cat([S_pos,    S_neg])
    p1 = prob_all[S_all == 1].mean()
    p0 = prob_all[S_all == 0].mean()
    dp_penalty = (p1 - p0).abs()

    penalty_w = 20.0
    loss = bce + fair_lambda * penalty_w * dp_penalty
    loss.backward()
    optimizer.step()

    return float(loss.item()), float(dp_penalty.item())

from sklearn.metrics import roc_auc_score, average_precision_score
from fairness import demographic_parity_gap

@torch.no_grad()
def evaluate(model, x, full_edge_index, pos_edges, S_node, num_neg=None):
    model.eval()

    if num_neg is None:
        num_neg = 4 * pos_edges.size(1)
    neg_edges = sample_negatives(full_edge_index, x.size(0), num_neg)

    pos_score, neg_score, _ = model(x, full_edge_index, pos_edges, neg_edges)

    # metrics: AUC / AP
    y_true  = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]).cpu().numpy()
    y_score = torch.sigmoid(torch.cat([pos_score, neg_score])).cpu().numpy()
    auc = roc_auc_score(y_true, y_score)
    ap  = average_precision_score(y_true, y_score)

    # DP on pos+neg (与训练一致)
    S_pos = edge_sensitive_attr(S_node, pos_edges)
    S_neg = edge_sensitive_attr(S_node, neg_edges)
    S_all = torch.cat([S_pos, S_neg])
    scores = torch.sigmoid(torch.cat([pos_score, neg_score]))
    dp = demographic_parity_gap(scores, S_all)

    return float(auc), float(ap), float(dp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='LocalSBM', choices=['Cora','Citeseer','Pubmed','LocalSBM'])
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--fair_lambda', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--results_dir', type=str, default='results')
    p.add_argument('--sbm_ratio', type=float, default=0.5)
    p.add_argument('--p_in', type=float, default=0.08)
    p.add_argument('--p_out', type=float, default=0.01)
    p.add_argument('--feat_dim', type=int, default=128)
    p.add_argument('--sensitive', type=str, default='degree', choices=['degree','eigen','betw'])
    args = p.parse_args()

    set_seed(args.seed)
    ds_kwargs = dict(ratio=args.sbm_ratio, p_in=args.p_in, p_out=args.p_out,
                     feat_dim=args.feat_dim, seed=args.seed, sensitive=args.sensitive)
    dataset, data = load_dataset(args.dataset, **ds_kwargs)

    model = LinkPredictor(dataset.num_features, args.hidden, dropout=args.dropout)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_pos, val_pos, test_pos = split_edges(data.edge_index)
    logs = []
    for epoch in tqdm(range(1, args.epochs+1)):
        loss, pen = train_epoch(model, data.x, data.edge_index, train_pos, data.S, optim, fair_lambda=args.fair_lambda)
        if epoch % 10 == 0 or epoch == args.epochs:
            auc_val, ap_val, dp_val = evaluate(model, data.x, data.edge_index, val_pos, data.S)
            logs.append({'epoch': epoch, 'loss': loss, 'penalty': pen, 'val_auc': auc_val, 'val_ap': ap_val, 'val_dp': dp_val})
    auc_test, ap_test, dp_test = evaluate(model, data.x, data.edge_index, test_pos, data.S)
    payload = {'task':'link_prediction','dataset':args.dataset,'hidden':args.hidden,'epochs':args.epochs,
               'fair_lambda':args.fair_lambda,'sensitive':args.sensitive,'sbm_ratio':args.sbm_ratio,
               'p_in':args.p_in,'p_out':args.p_out,'test_metrics':{'auc':auc_test,'ap':ap_test,'dp_gap':dp_test},
               'val_logs':logs}
    path = save_results(args.results_dir, payload, prefix=f"linkpred_{args.dataset}_lambda{args.fair_lambda}_{args.sensitive}")
    print(f"Saved: {path}  |  Test auc={auc_test:.4f} ap={ap_test:.4f} dp={dp_test:.4f}")

if __name__ == '__main__':
    main()

