# visualization.py
# Script for plotting training metrics

import matplotlib.pyplot as plt

def plot_training_metrics(train_losses, val_losses, 
                         train_accuracies, val_accuracies,
                         learning_rates):
    """
    Plot training/validation metrics and learning rate schedule
    
    Args:
        train_losses (list): Training loss values per epoch
        val_losses (list): Validation loss values per epoch
        train_accuracies (list): Training accuracy values per epoch
        val_accuracies (list): Validation accuracy values per epoch  
        learning_rates (list): Learning rate values per epoch
    """
    actual_epochs = len(train_losses)
    
    # Create figure with 3 subplots
    plt.figure(figsize=(15, 10))
    
    # Plot Loss Curves
    plt.subplot(2, 2, 1)
    plt.plot(range(1, actual_epochs + 1), train_losses, label='Train')
    plt.plot(range(1, actual_epochs + 1), val_losses, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training/Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy Curves
    plt.subplot(2, 2, 2)
    plt.plot(range(1, actual_epochs + 1), train_accuracies, label='Train')
    plt.plot(range(1, actual_epochs + 1), val_accuracies, label='Validation') 
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training/Validation Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Learning Rate Schedule
    plt.subplot(2, 2, 3)
    plt.plot(range(1, actual_epochs + 1), learning_rates)
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
