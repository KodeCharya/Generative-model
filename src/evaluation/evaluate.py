import torch
import torch.nn.functional as F

def evaluate_model(model, validation_data, metrics):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    num_batches = len(validation_data)

    with torch.no_grad():
        for batch in validation_data:
            inputs, labels = batch
            outputs = model(inputs)
            loss = compute_loss(outputs, labels)
            total_loss += loss.item()
            total_accuracy += compute_accuracy(outputs, labels)

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    metrics['loss'] = avg_loss
    metrics['accuracy'] = avg_accuracy

    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_accuracy:.4f}")

def compute_loss(outputs, labels):
    return F.cross_entropy(outputs, labels)

def compute_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy