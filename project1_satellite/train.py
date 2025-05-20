# Start training timer
start_time = time.time()

# Training loop
for epoch in range(num_epochs):
    # Start epoch timer
    epoch_start_time = time.time()

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)

# Training phase
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

train_loss = running_loss / len(train_loader)
train_acc = 100 * correct / total
train_losses.append(train_loss)
train_accuracies.append(train_acc)

# Validation phase
model.eval()
val_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_loss /= len(val_loader)
val_acc = 100 * correct / total
val_losses.append(val_loss)
val_accuracies.append(val_acc)

# Save best model
if val_acc > best_acc:
    best_acc = val_acc
    torch.save(model.state_dict(), 'best_model.pth')
    print(f"✨ Best model saved with validation accuracy: {val_acc:.2f}%")

# Calculate epoch time
epoch_time = time.time() - epoch_start_time

# Display training info
print(f"\nEpoch {epoch+1}/{num_epochs}")
print(f"⏱️  Time: {epoch_time:.2f}s")
print(f"📊 Train | Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
print(f"📊 Val   | Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
print(f"📈 LR: {current_lr:.6f}")
print("-" * 50)

# Update learning rate
scheduler.step()

# Calculate total training time
total_time = time.time() - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)

# Final summary
print("\n" + "=" * 50)
print("📋 Training Complete")
print(f"⏱️  Duration: {hours}h {minutes}m {seconds}s")
print(f"🏆 Best Val Accuracy: {best_acc:.2f}%")
print("=" * 50)