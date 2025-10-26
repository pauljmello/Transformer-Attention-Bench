import os
import time
from datetime import datetime

import pandas as pd
import torch

from cnn import CNN
from mlp import MLP
from visuals import plot_training_curves, plot_needle_haystack_performance


def create_experiment_folder():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join("results", timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def save_results_to_csv(results, experiment_dir):
    csv_path = os.path.join(experiment_dir, "experiment_results.csv")
    csv_data = []
    for model_name, metrics in results.items():
        architecture = "MLP" if model_name.startswith("MLP") else "CNN"
        attention_type = model_name.split("-", 1)[1] if "-" in model_name else "Control"
        training_time = metrics['training_time']
        num_epochs = len(metrics['train_losses'])
        for epoch in range(num_epochs):
            epoch_time = training_time * (epoch + 1) / num_epochs
            csv_data.append({'model_name': model_name, 'architecture': architecture, 'attention_type': attention_type, 'epoch': epoch + 1, 'train_loss': metrics['train_losses'][epoch], 'train_accuracy': metrics['train_accuracies'][epoch],
                             'test_loss': metrics['test_losses'][epoch], 'test_accuracy': metrics['test_accuracies'][epoch], 'cumulative_training_time': epoch_time})
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    print(f"Detailed epoch-by-epoch results saved to: {csv_path}")
    summary_csv_path = os.path.join(experiment_dir, "experiment_summary.csv")
    summary_data = []
    for model_name, metrics in results.items():
        architecture = "MLP" if model_name.startswith("MLP") else "CNN"
        attention_type = model_name.split("-", 1)[1] if "-" in model_name else "Control"
        summary_data.append({'model_name': model_name, 'architecture': architecture, 'attention_type': attention_type, 'final_test_accuracy': metrics['test_accuracies'][-1], 'max_test_accuracy': max(metrics['test_accuracies']),
                             'final_train_accuracy': metrics['train_accuracies'][-1], 'final_train_loss': metrics['train_losses'][-1], 'min_train_loss': min(metrics['train_losses']), 'total_training_time': metrics['training_time'],
                             'num_epochs': len(metrics['train_losses'])})
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary results saved to: {summary_csv_path}")


def compute_accuracy(logits, labels):
    pred_labels = torch.argmax(logits, dim=1)
    num_correct = (pred_labels == labels).sum()
    return num_correct


def build_models(configuration):
    models = {}
    optimizers = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for attention_type, attention_params, hidden_sizes in configuration['attention_configurations']:
        attention_name = attention_type or 'Control'
        mlp_model = MLP(input_size=configuration['input_feature_size'], hidden_size=hidden_sizes['mlp_hidden'], num_classes=configuration['number_of_classes'], attention_type=attention_type, attention_params=attention_params)
        cnn_model = CNN(input_channels=3, num_classes=configuration['number_of_classes'], hidden_size=hidden_sizes['cnn_hidden'], attention_type=attention_type, attention_params=attention_params)

        mlp_model = mlp_model.to(device)
        cnn_model = cnn_model.to(device)
        cnn_model = cnn_model.to(memory_format=torch.channels_last)

        mlp_name = f'MLP - {attention_name}'
        cnn_name = f'CNN - {attention_name}'
        models[mlp_name] = mlp_model
        models[cnn_name] = cnn_model
        optimizers[mlp_name] = torch.optim.AdamW(mlp_model.parameters(), lr=configuration['learning_rate'])
        optimizers[cnn_name] = torch.optim.AdamW(cnn_model.parameters(), lr=configuration['learning_rate'])
    return models, optimizers


def train_batch(model, optimizer, batch_inputs, batch_labels, device):
    batch_inputs = batch_inputs.to(device, non_blocking=True)
    batch_labels = batch_labels.to(device, non_blocking=True)
    optimizer.zero_grad()
    logits, _ = model(batch_inputs)
    loss = model.compute_loss(logits, batch_labels)
    loss.backward()
    optimizer.step()
    num_correct = compute_accuracy(logits.detach(), batch_labels)
    return loss.detach(), num_correct


def train_one_epoch(model, optimizer, train_loader, configuration):
    device = next(model.parameters()).device
    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0, device=device)
    total_samples = 0

    for batch_inputs, batch_labels in train_loader:
        batch_loss, num_correct = train_batch(model, optimizer, batch_inputs, batch_labels, device)
        total_loss += batch_loss
        total_correct += num_correct
        total_samples += batch_labels.size(0)

    average_loss = (total_loss / len(train_loader)).item()
    average_accuracy = (total_correct / total_samples).item()
    return average_loss, average_accuracy


def evaluate_on_dataset(model, test_loader, configuration):
    device = next(model.parameters()).device
    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0, device=device)
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            logits, _ = model(batch_inputs)
            loss = model.compute_loss(logits, batch_labels)
            num_correct = compute_accuracy(logits, batch_labels)
            total_loss += loss
            total_correct += num_correct
            total_samples += batch_labels.size(0)

    average_loss = (total_loss / len(test_loader)).item()
    average_accuracy = (total_correct / total_samples).item()
    return average_loss, average_accuracy


def train_and_validate_model(model, optimizer, train_loader, test_loader, configuration):
    model.train()
    training_losses = []
    training_accuracies = []
    testing_losses = []
    testing_accuracies = []
    for epoch in range(configuration['number_of_epochs']):
        train_loss, train_accuracy = train_one_epoch(model, optimizer, train_loader, configuration)
        testing_loss, testing_accuracy = evaluate_on_dataset(model, test_loader, configuration)
        training_losses.append(train_loss)
        training_accuracies.append(train_accuracy)
        testing_losses.append(testing_loss)
        testing_accuracies.append(testing_accuracy)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_accuracy:.4f}, Test Acc={testing_accuracy:.4f}")
    return training_losses, training_accuracies, testing_losses, testing_accuracies


def run_all_experiments(models, optimizers, train_loader, test_loader, configuration):
    results = {}
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        start_time = time.time()
        optimizer = optimizers[model_name]
        train_losses, train_accuracies, test_losses, test_accuracies = train_and_validate_model(model, optimizer, train_loader, test_loader, configuration)
        training_time = time.time() - start_time
        results[model_name] = {'train_losses': train_losses, 'train_accuracies': train_accuracies, 'test_losses': test_losses, 'test_accuracies': test_accuracies, 'training_time': training_time}
        print(f"{model_name} completed: {test_accuracies[-1]:.4f} accuracy in {training_time:.2f}s")
        print("-" * 50)
    return results


def summarize_results(results, experiment_directory):
    for architecture in ('MLP', 'CNN'):
        print(f"\n{architecture} Results:")
        for model_name in sorted(results):
            if model_name.startswith(architecture):
                final_accuracy = results[model_name]['test_accuracies'][-1]
                elapsed_time = results[model_name]['training_time']
                print(f"{model_name:20s}: {final_accuracy:.4f} acc, {elapsed_time:6.2f}s")
    plot_training_curves(results, experiment_directory)
    print(f"\nAll results saved to: {experiment_directory}")


def evaluate_needle_haystack(models, test_x, test_y, config, experiment_dir):
    if not hasattr(models, '__iter__'):
        return
    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for model_name, model in models.items():
        if not model_name.startswith('CNN'):
            continue
        accuracies = []
        for noise_level in noise_levels:
            test_batch = test_x[:config['training_batch_size']].clone()
            if noise_level > 0:
                noise = torch.randn_like(test_batch) * noise_level
                test_batch = test_batch + noise
            test_labels = test_y[:config['training_batch_size']]
            model.eval()
            with torch.no_grad():
                logits, _ = model.forward(test_batch)
                num_correct = compute_accuracy(logits, test_labels)
                accuracy = (num_correct / test_labels.size(0)).item()
            accuracies.append(accuracy)
        model_dir = os.path.join(experiment_dir, model_name.replace(' ', '_').replace('-', '_'))
        os.makedirs(model_dir, exist_ok=True)
        plot_needle_haystack_performance(accuracies, noise_levels, model_dir)
