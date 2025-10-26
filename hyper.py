import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from cnn import CNN
from mlp import MLP
from utils import compute_accuracy


@dataclass
class HyperConfig:
    model_type: str
    attention_type: Optional[str]
    learning_rate: float
    batch_size: int
    hidden_size: int
    attention_params: dict


SEARCH_SPACE = {
    'learning_rate': [0.001, 0.01, 0.003],
    'batch_size': [512],
    'hidden_size': [256, 512],
}

ATTENTION_PARAMS = {
    None: {},
    'self': {},
    'flex': {},
    'multi_head': {'num_heads': [2, 4, 8]},
    'linear': {'num_heads': [2, 4, 8], 'kernel_size': [16, 32, 64, 128]},
    'performer': {'num_heads': [2, 4, 8], 'num_features': [64, 128, 256, 512]},
    'linformer': {'num_heads': [2, 4, 8], 'k': [32, 64, 128, 256], 'max_seq_len': [512, 1024]},
    'kv_compression': {'num_heads': [2, 4, 8], 'compression_ratio': [0.25, 0.5, 0.75]},
    'sparse': {'num_heads': [2, 4, 8], 'block_size': [32, 64, 128]},
    'local': {'num_heads': [2, 4, 8], 'window_size': [16, 32, 64, 128]},
    'deepseek_mla': {'num_heads': [2, 4, 8]},
    'kimi_k2': {'num_heads': [2, 4, 8]},
}


def load_data_split(train_size=5000, val_size=1000):
    from data import load_cifar10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_x_full, train_y_full, _, _ = load_cifar10()

    indices = np.random.permutation(len(train_x_full))
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]

    train_x = torch.from_numpy(train_x_full[train_idx]).permute(0, 3, 1, 2).float().to(device)
    train_y = torch.from_numpy(train_y_full[train_idx]).long().to(device)
    val_x = torch.from_numpy(train_x_full[val_idx]).permute(0, 3, 1, 2).float().to(device)
    val_y = torch.from_numpy(train_y_full[val_idx]).long().to(device)

    return train_x, train_y, val_x, val_y


def create_model(config: HyperConfig):
    if config.model_type == 'MLP':
        return MLP(3072, config.hidden_size, 10, config.attention_type, config.attention_params)
    else:
        return CNN(3, 10, config.hidden_size, config.attention_type, config.attention_params)


def train_epoch(model, optimizer, x, y, batch_size):
    model.train()
    indices = torch.randperm(len(x))
    x, y = x[indices], y[indices]

    total_loss = 0.0
    total_correct = 0
    num_batches = len(x) // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size

        optimizer.zero_grad()
        logits, _ = model(x[start:end])
        loss = model.compute_loss(logits, y[start:end])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += compute_accuracy(logits, y[start:end]).item()

    return total_loss / num_batches, total_correct / len(x)


def evaluate(model, x, y, batch_size):
    model.eval()
    total_correct = 0

    with torch.no_grad():
        num_batches = len(x) // batch_size
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            logits, _ = model(x[start:end])
            total_correct += compute_accuracy(logits, y[start:end]).item()

    return total_correct / len(x)


def train_config(config: HyperConfig, train_x, train_y, val_x, val_y, epochs):
    model = create_model(config).to(train_x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

    for _ in range(epochs):
        train_epoch(model, optimizer, train_x, train_y, config.batch_size)

    val_acc = evaluate(model, val_x, val_y, config.batch_size)
    return val_acc


def generate_configs(attention_type: str, model_type: str, n_configs: int):
    configs = []
    for _ in range(n_configs):
        lr = float(np.random.choice(SEARCH_SPACE['learning_rate']))
        batch_size = int(np.random.choice(SEARCH_SPACE['batch_size']))
        hidden_size = int(np.random.choice(SEARCH_SPACE['hidden_size']))

        attn_params = {}
        for key, values in ATTENTION_PARAMS.get(attention_type, {}).items():
            value = np.random.choice(values)
            if isinstance(value, (np.integer, np.floating)):
                attn_params[key] = int(value) if isinstance(value, np.integer) else float(value)
            else:
                attn_params[key] = value

        configs.append(HyperConfig(model_type, attention_type, lr, batch_size, hidden_size, attn_params))

    return configs


def successive_halving(configs, train_x, train_y, val_x, val_y, rounds=3):
    current_configs = configs
    epochs = 3

    for round_idx in range(rounds):
        print(f"  Round {round_idx + 1}/{rounds}: {len(current_configs)} configs, {epochs} epochs")
        results = []

        for i, cfg in enumerate(current_configs):
            val_acc = train_config(cfg, train_x, train_y, val_x, val_y, epochs)
            results.append((cfg, val_acc))
            print(f"    [{i + 1}/{len(current_configs)}] Val Acc: {val_acc:.4f}")

        results.sort(key=lambda x: x[1], reverse=True)

        if round_idx < rounds - 1:
            n_keep = max(1, len(results) // 2)
            current_configs = [cfg for cfg, _ in results[:n_keep]]
            epochs += 2
        else:
            return results[0]


def run_sweep(attention_types=None, model_types=None, n_configs=20, save_dir='sweep_results'):
    if attention_types is None:
        attention_types = list(ATTENTION_PARAMS.keys())
    if model_types is None:
        model_types = ['MLP', 'CNN']

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    train_x, train_y, val_x, val_y = load_data_split()

    results = {}
    for attention_type in attention_types:
        for model_type in model_types:
            key = f"{model_type}_{attention_type or 'Control'}"
            print(f"\n{'=' * 50}")
            print(f"Sweeping: {key}")
            print('=' * 50)

            configs = generate_configs(attention_type, model_type, n_configs)
            best_config, best_acc = successive_halving(configs, train_x, train_y, val_x, val_y)
            results[key] = {'config': asdict(best_config), 'val_acc': best_acc}

            print(f"\nBest {key}: {best_acc:.4f}\n")

    with open(save_path / f'summary_{timestamp}.txt', 'w') as f:
        for key, data in sorted(results.items(), key=lambda x: x[1]['val_acc'], reverse=True):
            f.write(f"{key}: {data['val_acc']:.4f}\n")
            for k, v in data['config'].items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

    print(f"\nResults saved to: {save_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Hyperparameter sweep')
    parser.add_argument('--attention', nargs='+', default=['all'], help='Attention types to sweep')
    parser.add_argument('--model', nargs='+', default=['both'], help='Model types to sweep')
    parser.add_argument('--n-configs', type=int, default=20, help='Number of configs per sweep')
    parser.add_argument('--save-dir', type=str, default='sweep_results', help='Directory to save results')
    args = parser.parse_args()

    attention_types = list(ATTENTION_PARAMS.keys()) if 'all' in args.attention else [None if a == 'none' else a for a in args.attention]
    model_types = ['MLP', 'CNN'] if 'both' in args.model else args.model

    print(f"Sweep: {len(attention_types)} attention types × {len(model_types)} models × {args.n_configs} configs")
    run_sweep(attention_types, model_types, args.n_configs, args.save_dir)
