import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_model_results(model_results, model_type, experiment_dir, linestyle='-', colors=None):
    if colors is None:
        colors = {'MLP': {'accuracy': 'skyblue', 'time': 'lightcoral'}, 'CNN': {'accuracy': 'lightgreen', 'time': 'gold'}}.get(model_type, {'accuracy': 'steelblue', 'time': 'coral'})

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot training curves
    for model_name, metrics in model_results.items():
        epochs = range(len(metrics['train_losses']))
        attention_name = model_name.replace(f'{model_type} - ', '')

        axes[0, 0].plot(epochs, metrics['train_losses'], label=attention_name, linewidth=2, linestyle=linestyle)
        axes[0, 1].plot(epochs, metrics['test_accuracies'], label=attention_name, linewidth=2, linestyle=linestyle)

    # Configure loss plot
    axes[0, 0].set_title(f'{model_type} Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Configure accuracy plot
    axes[0, 1].set_title(f'{model_type} Test Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Extract data for bar charts
    model_names = [name.replace(f'{model_type} - ', '') for name in model_results.keys()]
    accuracies = [result['test_accuracies'][-1] for result in model_results.values()]
    times = [result['training_time'] for result in model_results.values()]

    # Final accuracy bar chart
    axes[1, 0].bar(model_names, accuracies, color=colors['accuracy'], alpha=0.8)
    axes[1, 0].set_title(f'{model_type} Final Test Accuracy')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)

    # Training time bar chart
    axes[1, 1].bar(model_names, times, color=colors['time'], alpha=0.8)
    axes[1, 1].set_title(f'{model_type} Training Time')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(experiment_dir, f"{model_type.lower()}_results.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"{model_type} results saved to: {output_path}")


def plot_training_curves(results, experiment_dir):
    from utils import save_results_to_csv
    mlp_results = {k: v for k, v in results.items() if k.startswith('MLP')}
    cnn_results = {k: v for k, v in results.items() if k.startswith('CNN')}

    if mlp_results:
        plot_model_results(mlp_results, 'MLP', experiment_dir, linestyle='-')

    if cnn_results:
        plot_model_results(cnn_results, 'CNN', experiment_dir, linestyle='--')

    save_results_to_csv(results, experiment_dir)


def plot_attention_matrix(attention_weights, head_idx=0, experiment_dir="."):
    if attention_weights is None:
        return

    attention = attention_weights[0, head_idx] if len(attention_weights.shape) > 3 else attention_weights[0]
    if hasattr(attention, 'cpu'):
        attention = attention.cpu().detach().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    im = ax.imshow(attention, cmap='Blues', aspect='auto')
    ax.set_title(f'Attention Matrix (Head {head_idx})')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    plt.colorbar(im, ax=ax, label='Attention Weight')
    plt.tight_layout()

    attention_path = os.path.join(experiment_dir, f'attention_matrix_head_{head_idx}.png')
    plt.savefig(attention_path, bbox_inches='tight')
    plt.close()
    print(f"Attention matrix saved to: {attention_path}")


def plot_needle_haystack_performance(accuracies, noise_levels, experiment_dir="."):
    plt.figure(figsize=(14, 6))
    plt.plot(noise_levels, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='Random baseline (10 classes)')
    plt.title('Robustness to Input Noise')
    plt.xlabel('Noise Level (std)')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    noise_path = os.path.join(experiment_dir, 'noise_robustness.png')
    plt.savefig(noise_path, bbox_inches='tight')
    plt.close()
    print(f"Noise robustness plot saved to: {noise_path}")


def visualize_filters(weights, title='Learned Filters', experiment_dir="."):
    if len(weights.shape) != 4:
        return

    num_filters = min(16, weights.shape[-1])
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))

    for i in range(num_filters):
        row = i // 4
        col = i % 4

        filter_img = weights[:, :, :, i]
        if filter_img.shape[2] == 3:  # RGB
            filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())
            axes[row, col].imshow(filter_img)
        else:  # Grayscale
            axes[row, col].imshow(filter_img[:, :, 0], cmap='gray')

        axes[row, col].set_title(f'Filter {i + 1}')
        axes[row, col].axis('off')

    plt.suptitle(title)
    plt.tight_layout()

    filters_path = os.path.join(experiment_dir, 'learned_filters.png')
    plt.savefig(filters_path, bbox_inches='tight')
    plt.close()
    print(f"Learned filters saved to: {filters_path}")


def visualize_attention_patterns(models, sample_inputs, experiment_dir):
    for model_name, model in models.items():
        model.eval()
        with torch.no_grad():
            _, attention_weights = model.forward(sample_inputs)
        if attention_weights is not None:
            attn_shape = attention_weights.shape
            seq_len = attn_shape[-1]
            if seq_len < 4:
                continue
            model_dir = os.path.join(experiment_dir, model_name.replace(' ', '_').replace('-', '_'))
            os.makedirs(model_dir, exist_ok=True)
            if len(attention_weights.shape) > 3:
                num_heads = attention_weights.shape[1]
                for head_idx in range(min(num_heads, 4)):
                    plot_attention_matrix(attention_weights, head_idx, model_dir)
            else:
                plot_attention_matrix(attention_weights, 0, model_dir)


def visualize_cnn_filters(models, experiment_dir):
    for model_name, model in models.items():
        if not model_name.startswith('CNN'):
            continue
        model_dir = os.path.join(experiment_dir, model_name.replace(' ', '_').replace('-', '_'))
        os.makedirs(model_dir, exist_ok=True)

        conv1_weights = model.block1[0].weight.data.cpu().numpy()
        conv1_weights = conv1_weights.transpose(2, 3, 1, 0)
        visualize_filters(conv1_weights, title=f'{model_name} - Block1 Filters', experiment_dir=model_dir)

        conv2_weights = model.block2[0].weight.data.cpu().numpy()
        conv2_mean = np.abs(conv2_weights).mean(axis=1)
        conv2_mean = conv2_mean.transpose(1, 2, 0)
        conv2_mean = np.stack([conv2_mean, conv2_mean, conv2_mean], axis=2)
        visualize_filters(conv2_mean, title=f'{model_name} - Block2 Filters (Averaged)', experiment_dir=model_dir)
