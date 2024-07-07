import torch
from transformers import AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import os

# Function to initialize the model and optimizer


def initialize_model(model_name, num_labels, learning_rate, device):
    """
    Initializes the model and optimizer.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    return model, optimizer

# Function to create the learning rate scheduler


def get_scheduler(optimizer, train_loader, epochs):
    """
    Creates the learning rate scheduler.
    """
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    return scheduler

# Function to calculate accuracy


def calculate_accuracy(predictions, labels):
    """
    Calculates accuracy.
    """
    _, predicted_labels = torch.max(predictions, 1)
    correct_predictions = (predicted_labels == labels).sum().item()
    return correct_predictions

# Function to validate the model on the validation set


def validate_model(model, data_loader, loss_function, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_correct = 0
    total_steps = 0
    total_examples = 0
    total_loss = 0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for _, data in enumerate(data_loader, 0):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_function(outputs.logits, labels)
            total_loss += loss.item()

            total_correct += calculate_accuracy(outputs.logits, labels)
            predicted_labels.extend(
                outputs.logits.argmax(dim=1).cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

            total_steps += 1
            total_examples += labels.size(0)

    avg_loss = total_loss / total_steps
    avg_accuracy = (total_correct * 100) / total_examples
    print(f"Validation Loss: {avg_loss}")
    print(f"Validation Accuracy: {avg_accuracy}\n")

    return true_labels, predicted_labels, avg_accuracy

# Function to train the model


def train_model(model, train_loader, val_loader, optimizer, scheduler, loss_function, device, epochs, save_path, tokenizer):
    """
    Trains the model and validates after each epoch.
    """
    for epoch in range(epochs):
        model.train()
        total_correct = 0
        total_steps = 0
        total_examples = 0
        total_loss = 0

        for batch, data in enumerate(train_loader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_function(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_correct += calculate_accuracy(outputs.logits, labels)
            total_steps += 1
            total_examples += labels.size(0)

            if batch % 100 == 0:
                step_loss = total_loss / total_steps
                step_accuracy = (total_correct * 100) / total_examples

                test_examples(model, tokenizer, device)

                print(
                    f"Batch {batch} of epoch {epoch+1} complete. Loss: {step_loss} Accuracy: {step_accuracy}")

        avg_loss = total_loss / total_steps
        avg_accuracy = (total_correct * 100) / total_examples
        print(f"Training Loss: {avg_loss}")
        print(f"Training Accuracy: {avg_accuracy}\n")

        checkpoint_path = os.path.join(save_path, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        validate_model(model, val_loader, loss_function, device)


def test_examples(model, tokenizer, device):

    rev_1 = "This product is amazing!"
    rev_2 = "I would not recommend this product."
    rev_3 = "The product is okay, but not great."

    model.eval()
    with torch.no_grad():
        for review in [rev_1, rev_2, rev_3]:
            encoded = tokenizer(review, return_tensors='pt',
                                padding=True, truncation=True, max_length=256)
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            predicted_label = torch.argmax(outputs.logits).item() + 1
            print(f"Review: {review}")
            print(f"Predicted Star Rating: {predicted_label}\n")

    model.train()
