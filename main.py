import torch
import torch.nn.attention.flex_attention

from data import load_dataset
from utils import build_models, create_experiment_folder, evaluate_needle_haystack, run_all_experiments, summarize_results
from visuals import visualize_attention_patterns, visualize_cnn_filters


def get_configuration():
    training_batch_size = 512
    number_of_epochs = 5
    learning_rate = 0.001
    input_feature_size = 32 * 32 * 3
    number_of_classes = 10

    # Hyperparameters selected from hyper.py parameter sweep
    attention_configurations = [
        [None, {}, {'mlp_hidden': 512, 'cnn_hidden': 512}],
        ['self', {}, {'mlp_hidden': 256, 'cnn_hidden': 512}],
        # ['flex', {}, {'mlp_hidden': 512, 'cnn_hidden': 512}],  # C++ compiler error
        ['multi_head', {'num_heads': 2}, {'mlp_hidden': 256, 'cnn_hidden': 512}],
        ['linear', {'num_heads': 4, 'kernel_size': 128}, {'mlp_hidden': 256, 'cnn_hidden': 256}],
        ['performer', {'num_heads': 8, 'num_features': 64}, {'mlp_hidden': 256, 'cnn_hidden': 512}],
        ['linformer', {'num_heads': 2, 'k': 64, 'max_seq_len': 512}, {'mlp_hidden': 256, 'cnn_hidden': 512}],
        ['kv_compression', {'num_heads': 4, 'compression_ratio': 0.75}, {'mlp_hidden': 256, 'cnn_hidden': 256}],
        ['sparse', {'num_heads': 2, 'block_size': 32}, {'mlp_hidden': 256, 'cnn_hidden': 512}],
        ['local', {'num_heads': 2, 'window_size': 16}, {'mlp_hidden': 256, 'cnn_hidden': 512}],
        ['deepseek_mla', {'num_heads': 8}, {'mlp_hidden': 256, 'cnn_hidden': 512}],
        ['kimi_k2', {'num_heads': 4}, {'mlp_hidden': 256, 'cnn_hidden': 256}],
    ]

    return {'training_batch_size': training_batch_size, 'number_of_epochs': number_of_epochs, 'learning_rate': learning_rate, 'input_feature_size': input_feature_size, 'number_of_classes': number_of_classes,
            'attention_configurations': attention_configurations}


def main():
    configuration = get_configuration()
    experiment_directory = create_experiment_folder()
    print(f"Experiment directory: {experiment_directory}")

    train_loader, test_loader = load_dataset(batch_size=configuration['training_batch_size'])
    models, optimizers = build_models(configuration)
    results = run_all_experiments(models, optimizers, train_loader, test_loader, configuration)
    summarize_results(results, experiment_directory)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    testing_inputs = test_loader.dataset.tensors[0].to(device)
    testing_labels = test_loader.dataset.tensors[1].to(device)

    print("\nVisualizing attention patterns...")
    visualize_attention_patterns(models, testing_inputs[:configuration['training_batch_size']], experiment_directory)

    print("\nEvaluating robustness to noise...")
    evaluate_needle_haystack(models, testing_inputs, testing_labels, configuration, experiment_directory)

    print("\nVisualizing learned CNN filters...")
    visualize_cnn_filters(models, experiment_directory)

    print(f"\nAll visualizations saved to: {experiment_directory}")


if __name__ == "__main__":
    main()
