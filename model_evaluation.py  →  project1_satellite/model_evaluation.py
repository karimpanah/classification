# model_evaluation.py
# Script for evaluating trained classification models

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, precision_score, 
                           recall_score, f1_score, accuracy_score)
from collections import Counter

def evaluate_classifier(model, test_loader, criterion, device, class_names, num_display_images=10):
    """
    Evaluate model performance on test dataset
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test dataset
        criterion: Loss function
        device: Computation device (cuda/cpu)
        class_names: List of class names
        num_display_images: Number of misclassified samples to display
    
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    test_loss = 0.0
    all_labels = []
    all_preds = []
    misclassified_samples = []

    # Evaluation loop
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
            # Collect misclassified samples
            incorrect = (predicted != labels)
            if incorrect.any():
                idxs = incorrect.nonzero(as_tuple=True)[0]
                for idx in idxs:
                    misclassified_samples.append({
                        'image': images[idx].cpu(),
                        'true': labels[idx].item(),
                        'pred': predicted[idx].item()
                    })

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    test_loss /= len(test_loader)

    # Calculate metrics
    metrics = {
        'test_loss': test_loss,
        'overall_accuracy': accuracy_score(all_labels, all_preds),
        'overall_precision': precision_score(all_labels, all_preds, average='macro'),
        'overall_recall': recall_score(all_labels, all_preds, average='macro'),
        'overall_f1': f1_score(all_labels, all_preds, average='macro'),
        'class_precision': precision_score(all_labels, all_preds, average=None),
        'class_recall': recall_score(all_labels, all_preds, average=None),
        'class_f1': f1_score(all_labels, all_preds, average=None)
    }

    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Test Loss: {metrics['test_loss']:.4f}")
    
    print("\nOverall Metrics:")
    print(f"Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Precision: {metrics['overall_precision']:.4f}")
    print(f"Recall: {metrics['overall_recall']:.4f}")
    print(f"F1 Score: {metrics['overall_f1']:.4f}")

    print("\nPer-Class Metrics:")
    for i, name in enumerate(class_names):
        print(f"\n{name}:")
        print(f"Precision: {metrics['class_precision'][i]:.4f}")
        print(f"Recall: {metrics['class_recall'][i]:.4f}")
        print(f"F1: {metrics['class_f1'][i]:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Show misclassified examples
    if misclassified_samples:
        num_samples = min(num_display_images, len(misclassified_samples))
        fig, axes = plt.subplots(2, num_samples//2, figsize=(15, 6))
        axes = axes.ravel()

        # ImageNet normalization parameters
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        for i in range(num_samples):
            sample = misclassified_samples[i]
            img = sample['image'].permute(1, 2, 0).numpy()
            img = img * std + mean  # Denormalize
            img = np.clip(img, 0, 1)
            
            true_label = class_names[sample['true']]
            pred_label = class_names[sample['pred']]

            axes[i].imshow(img)
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}')
            axes[i].axis('off')

        plt.suptitle('Misclassified Samples', y=1.05)
        plt.tight_layout()
        plt.show()

    # Class distribution
    label_counts = Counter(all_labels)
    print("\nTest Set Class Distribution:")
    for class_idx, count in sorted(label_counts.items()):
        print(f"{class_names[class_idx]} ({class_idx}): {count} samples")

    metrics.update({
        'confusion_matrix': cm,
        'label_counts': label_counts
    })

    return metrics
