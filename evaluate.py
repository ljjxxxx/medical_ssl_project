import os
import torch
import torch.nn as nn
import numpy as np
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg for server environments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from models import ChestXrayClassifier
from data_utils import create_dataloaders

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, test_loader, device, disease_labels, output_dir="./evaluation_results"):
    """
    Evaluates the model on the test dataset and saves results.
    
    Args:
        model: The trained classifier model
        test_loader: DataLoader for the test dataset
        device: Device to run evaluation on
        disease_labels: List of disease names
        output_dir: Directory to save evaluation results
    
    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info("Evaluating model on test set")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            logits = outputs['logits']

            # Calculate probabilities and predictions
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_logits = np.vstack(all_logits)
    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    metrics = {}
    metrics['accuracy'] = accuracy_score(all_labels.flatten(), all_preds.flatten())
    metrics['precision'] = precision_score(all_labels.flatten(), all_preds.flatten(), zero_division=0)
    metrics['recall'] = recall_score(all_labels.flatten(), all_preds.flatten(), zero_division=0)
    metrics['f1'] = f1_score(all_labels.flatten(), all_preds.flatten(), zero_division=0)

    disease_metrics = {}
    avg_auc = 0
    num_diseases = len(disease_labels)

    for i, disease in enumerate(disease_labels):
        disease_metrics[disease] = {
            'precision': precision_score(all_labels[:, i], all_preds[:, i], zero_division=0),
            'recall': recall_score(all_labels[:, i], all_preds[:, i], zero_division=0),
            'f1': f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        }

        try:
            if np.any(all_labels[:, i] == 1):
                auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                disease_metrics[disease]['auc'] = auc
                avg_auc += auc
            else:
                disease_metrics[disease]['auc'] = float('nan')
        except ValueError:
            disease_metrics[disease]['auc'] = float('nan')

    metrics['avg_auc'] = avg_auc / num_diseases

    # Print overall metrics
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {metrics['precision']:.4f}")
    logger.info(f"Test Recall: {metrics['recall']:.4f}")
    logger.info(f"Test F1 Score: {metrics['f1']:.4f}")
    logger.info(f"Test Average AUC: {metrics['avg_auc']:.4f}")

    # Print per-disease metrics
    logger.info("Metrics for each disease:")
    for disease, metric in disease_metrics.items():
        logger.info(f"{disease}:")
        logger.info(f"  Precision: {metric['precision']:.4f}")
        logger.info(f"  Recall: {metric['recall']:.4f}")
        logger.info(f"  F1 Score: {metric['f1']:.4f}")
        logger.info(f"  AUC: {metric['auc']:.4f}")

    # Save metrics to file
    metrics_path = output_path / 'test_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("Test Set Evaluation Metrics:\n\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Overall Precision: {metrics['precision']:.4f}\n")
        f.write(f"Overall Recall: {metrics['recall']:.4f}\n")
        f.write(f"Overall F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"Average AUC: {metrics['avg_auc']:.4f}\n\n")

        f.write("Metrics for each disease:\n")
        for disease, metric in disease_metrics.items():
            f.write(f"{disease}:\n")
            f.write(f"  Precision: {metric['precision']:.4f}\n")
            f.write(f"  Recall: {metric['recall']:.4f}\n")
            f.write(f"  F1 Score: {metric['f1']:.4f}\n")
            f.write(f"  AUC: {metric['auc']:.4f}\n\n")
    
    # Create confusion matrices and ROC curves
    plt.figure(figsize=(8, 6))
    plt.bar(disease_labels, [disease_metrics[d]['f1'] for d in disease_labels])
    plt.xticks(rotation=90)
    plt.ylabel('F1 Score')
    plt.title('F1 Score by Disease')
    plt.tight_layout()
    plt.savefig(output_path / 'disease_f1_scores.png')
    plt.close()
    
    # Create AUC chart if available
    plt.figure(figsize=(8, 6))
    auc_values = []
    valid_diseases = []
    for disease in disease_labels:
        if not np.isnan(disease_metrics[disease]['auc']):
            auc_values.append(disease_metrics[disease]['auc'])
            valid_diseases.append(disease)
    
    if auc_values:
        plt.bar(valid_diseases, auc_values)
        plt.xticks(rotation=90)
        plt.ylabel('AUC')
        plt.title('AUC by Disease')
        plt.tight_layout()
        plt.savefig(output_path / 'disease_auc_scores.png')
    plt.close()
    
    logger.info(f"Evaluation results saved to {output_path}")
    
    return {
        'overall': metrics,
        'per_disease': disease_metrics
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate Chest X-ray Disease Classification Model')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/best_classifier_model.pt',
                        help='Path to the model checkpoint file')
    parser.add_argument('--data-dir', type=str, default='./datasets/chestxray14',
                        help='Path to the dataset directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run evaluation on (cuda, mps, or cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes for data loading')
    parser.add_argument('--strict-loading', action='store_true',
                        help='Use strict state_dict loading (set to False to skip non-matching keys)')
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else 
                             "mps" if torch.backends.mps.is_available() else 
                             "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return
    
    # Load data
    logger.info(f"Loading dataset from {args.data_dir}")
    try:
        dataloaders_with_labels = create_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        test_loader = dataloaders_with_labels['supervised']['test']
        disease_labels = dataloaders_with_labels['disease_labels']
        
        num_classes = len(disease_labels)
        logger.info(f"Loaded dataset with {num_classes} disease classes")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Create and load model
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    try:
        # Changed from resnet18 to resnet50 to match training architecture
        model = ChestXrayClassifier(
            backbone_name='resnet50',  # 使用ResNet50作为骨干网络
            pretrained=False,
            num_classes=num_classes
        ).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Use strict=False if needed for partial loading
        strict_loading = args.strict_loading
        logger.info(f"Using strict loading: {strict_loading}")
        
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict_loading)
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Evaluate model
    try:
        evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            disease_labels=disease_labels,
            output_dir=args.output_dir
        )
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    main()