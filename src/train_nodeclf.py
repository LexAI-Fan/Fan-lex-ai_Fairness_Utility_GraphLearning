import argparse, os
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from utils import set_seed, save_results
from data import load_dataset
from models import NodeClassifier
from fairness import demographic_parity_gap, equalized_odds_gap, corr_with_sensitive
def train_epoch(model, data, optimizer, fair_lambda=0.0):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    ce = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    probs = torch.softmax(logits, dim=-1)[:, 1]
    mask = data.train_mask
    p1 = probs[mask & (data.S == 1)].mean()
    p0 = probs[mask & (data.S == 0)].mean()
    dp_pen = (p1 - p0).abs()
    loss = ce + fair_lambda * dp_pen
    loss.backward()
    optimizer.step()
    return float(ce.item()), float(dp_pen.item())
@torch.no_grad()
def evaluate(model, data, split='val', pos_class=1):
    model.eval()
    logits = model(data.x, data.edge_index)
    probs = F.softmax(logits, dim=-1)[:, pos_class]
    preds = logits.argmax(dim=-1)
    mask = data.val_mask if split == 'val' else (data.test_mask if split == 'test' else data.train_mask)
    acc = accuracy_score(data.y[mask].cpu(), preds[mask].cpu())
    y_true_bin = (data.y == pos_class).long()
    dp = demographic_parity_gap(probs[mask], data.S[mask])
    eo = equalized_odds_gap(probs[mask], y_true_bin[mask], data.S[mask])
    return acc, dp, eo
@torch.no_grad()
def evaluate(model, data, split='val', pos_class=1):
    model.eval()
    logits = model(data.x, data.edge_index)
    probs = F.softmax(logits, dim=-1)[:, pos_class]
    preds = logits.argmax(dim=-1)
    mask = data.val_mask if split == 'val' else (data.test_mask if split == 'test' else data.train_mask)
    acc = accuracy_score(data.y[mask].cpu(), preds[mask].cpu())
    y_true_bin = (data.y == pos_class).long()
    dp = demographic_parity_gap(probs[mask], data.S[mask])
    eo = equalized_odds_gap(probs[mask], y_true_bin[mask], data.S[mask])
    return acc, dp, eo
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
    # LocalSBM knobs
    p.add_argument('--sbm_ratio', type=float, default=0.5, help='Block A fraction, e.g., 0.7 => 70/30')
    p.add_argument('--p_in', type=float, default=0.08)
    p.add_argument('--p_out', type=float, default=0.01)
    p.add_argument('--feat_dim', type=int, default=128)
    p.add_argument('--sensitive', type=str, default='degree', choices=['degree','eigen','betw'])
    args = p.parse_args()

    set_seed(args.seed)
    ds_kwargs = dict(ratio=args.sbm_ratio, p_in=args.p_in, p_out=args.p_out,
                     feat_dim=args.feat_dim, seed=args.seed, sensitive=args.sensitive)
    dataset, data = load_dataset(args.dataset, **ds_kwargs)

    model = NodeClassifier(dataset.num_features, args.hidden, getattr(dataset, 'num_classes', 2), dropout=args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logs = []
    for epoch in tqdm(range(1, args.epochs + 1)):
        ce, dp_pen = train_epoch(model, data, optimizer, args.fair_lambda)
        if epoch % 10 == 0 or epoch == args.epochs:
            acc_val, dp_val, eo_val = evaluate(model, data, split='val')
            logs.append({'epoch': epoch, 'ce': ce, 'dp_pen': dp_pen, 'val_acc': acc_val, 'val_dp': dp_val, 'val_eo': eo_val})
    acc_test, dp_test, eo_test = evaluate(model, data, split='test')
    payload = {
        'task': 'node_classification',
        'dataset': args.dataset,
        'hidden': args.hidden,
        'epochs': args.epochs,
        'fair_lambda': args.fair_lambda,
        'sensitive': args.sensitive,
        'sbm_ratio': args.sbm_ratio, 'p_in': args.p_in, 'p_out': args.p_out,
        'test_metrics': {'accuracy': acc_test, 'dp_gap': dp_test, 'eo_gap': eo_test},
        'val_logs': logs
    }
    path = save_results(args.results_dir, payload, prefix=f"nodeclf_{args.dataset}_lambda{args.fair_lambda}_{args.sensitive}")
    print(f"Saved: {path}  |  Test acc={acc_test:.4f} dp={dp_test:.4f} eo={eo_test:.4f}")
if __name__ == '__main__':
    main()
