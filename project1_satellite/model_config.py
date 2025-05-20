# Hyperparameters
batch_size = 64  # Batch size
num_workers = 2  # Number of workers for data loading

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

print('Dataloaders created successfully!')

# Load EfficientNet-B0
model = models.efficientnet_b0(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Modify classifier for 10 classes
# In EfficientNet, the classifier is a Sequential with a Linear layer
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)

# Unfreeze classifier layer
for param in model.classifier[1].parameters():
    param.requires_grad = True

# Unfreeze last convolutional layer
# In EfficientNet-B0, convolutional layers are in 'features'
# features[-1] is the last convolutional block
for param in model.features[-1].parameters():
    param.requires_grad = True

# Verify changes
print("Classifier:")
print(model.classifier)
print("\nLast features block:")
print(model.features[-1])

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

print('Model setup complete!')

# Training metrics storage
learning_rates = []  # Stores learning rate at each epoch
train_losses = []    # Training loss values
val_losses = []      # Validation loss values
train_accuracies = [] # Training accuracy
val_accuracies = []   # Validation accuracy
best_acc = 0.0        # Track best validation accuracy
num_epochs = 50       # Total training epochs