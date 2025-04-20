import os
import json
import matplotlib.pyplot as plt
import matplotlib
import argparse
from pathlib import Path

# Set the Agg backend to ensure plots can be generated on servers without GUI
matplotlib.use('Agg')


def regenerate_ssl_plot(json_file, output_dir):
    """
    Regenerate the SSL training curve with English labels
    """
    print(f"Reading SSL training data from {json_file}")
    try:
        with open(json_file, 'r') as f:
            train_state = json.load(f)

        train_losses = train_state.get('train_losses', [])
        val_losses = train_state.get('val_losses', [])
        best_val_loss = train_state.get('best_val_loss', float('inf'))

        print(f"Found {len(train_losses)} training loss points and {len(val_losses)} validation loss points")

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('SSL Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save the plot
        output_path = Path(output_dir) / 'ssl_training_curve.png'
        plt.savefig(output_path)
        plt.close()

        print(f"SSL training curve saved to {output_path}")
        return True

    except Exception as e:
        print(f"Error regenerating SSL plot: {e}")
        return False


def regenerate_classifier_plot(json_file, output_dir):
    """
    Regenerate the classifier training curve with English labels
    """
    print(f"Reading classifier training data from {json_file}")
    try:
        with open(json_file, 'r') as f:
            train_state = json.load(f)

        train_losses = train_state.get('train_losses', [])
        val_losses = train_state.get('val_losses', [])
        val_metrics = train_state.get('val_metrics', [])
        best_val_metric = train_state.get('best_val_metric', 0)

        print(f"Found {len(train_losses)} training loss points and {len(val_metrics)} validation metric points")

        # Create the plot with multiple subplots
        plt.figure(figsize=(12, 8))

        # Plot 1: Training and Validation Loss
        plt.subplot(2, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: F1 Score and Accuracy
        plt.subplot(2, 2, 2)
        if val_metrics:
            plt.plot([m.get('f1', 0) for m in val_metrics], label='F1 Score')
            plt.plot([m.get('accuracy', 0) for m in val_metrics], label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('F1 and Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 3: Precision and Recall
        plt.subplot(2, 2, 3)
        if val_metrics:
            plt.plot([m.get('precision', 0) for m in val_metrics], label='Precision')
            plt.plot([m.get('recall', 0) for m in val_metrics], label='Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Precision and Recall')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 4: AUC
        plt.subplot(2, 2, 4)
        if val_metrics:
            plt.plot([m.get('auc', 0) for m in val_metrics], label='AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('Area Under ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot
        output_path = Path(output_dir) / 'classifier_training_curve.png'
        plt.savefig(output_path)
        plt.close()

        print(f"Classifier training curve saved to {output_path}")
        return True

    except Exception as e:
        print(f"Error regenerating classifier plot: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Regenerate training curve plots with English labels')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory containing training state JSON files')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save regenerated plots (defaults to checkpoint-dir)')
    parser.add_argument('--ssl-only', action='store_true',
                        help='Only regenerate SSL training curve')
    parser.add_argument('--classifier-only', action='store_true',
                        help='Only regenerate classifier training curve')
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir if args.output_dir else args.checkpoint_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for training state files
    ssl_json = checkpoint_dir / 'ssl_train_state.json'
    classifier_json = checkpoint_dir / 'classifier_train_state.json'

    if not args.classifier_only and ssl_json.exists():
        regenerate_ssl_plot(ssl_json, output_dir)
    elif not args.classifier_only:
        print(f"SSL training state file not found at {ssl_json}")

    if not args.ssl_only and classifier_json.exists():
        regenerate_classifier_plot(classifier_json, output_dir)
    elif not args.ssl_only:
        print(f"Classifier training state file not found at {classifier_json}")

    print("Plot regeneration complete")


if __name__ == '__main__':
    main()