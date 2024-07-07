import torch
from transformers import AutoTokenizer
from data_preparation import load_and_prepare_data, split_data
from dataset import create_data_loaders
from model_training import initialize_model, get_scheduler, train_model, validate_model

# Configuration settings
DATA_URL = './data/reviews_comments_stars.csv'
MODEL_NAME = 'roberta-base'  # or 'microsoft/deberta-v3-base'
MAX_LENGTH = 256
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 2e-5
SAVE_PATH = '/content/checkpoints/'
# Use GPU if available, otherwise use CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load and prepare the data
df = load_and_prepare_data(DATA_URL)
# Split the data into training, validation, and test sets
train_df, val_df, test_df = split_data(df)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(
    train_df, val_df, test_df, tokenizer, MAX_LENGTH, BATCH_SIZE)

# Initialize the model and optimizer
model, optimizer = initialize_model(MODEL_NAME, num_labels=len(
    df['stars'].unique()), learning_rate=LEARNING_RATE, device=DEVICE)
# Initialize the learning rate scheduler
scheduler = get_scheduler(optimizer, train_loader, EPOCHS)

# Calculate class weights
star_distribution = train_df['stars'].value_counts().sort_index().values
weights = 1.0 / star_distribution
weights = weights / weights.sum()  # Normalize weights
weights = torch.tensor(weights, dtype=torch.float32).to(
    DEVICE)  # Convert to tensor and move to the device

# Define the loss function with class weights
loss_function = torch.nn.CrossEntropyLoss(weight=weights)

# Train the model and validate
train_model(model, train_loader, val_loader, optimizer,
            scheduler, loss_function, DEVICE, EPOCHS, SAVE_PATH, tokenizer)

# Evaluate on the test set
print("Evaluation on test set")
validate_model(model, test_loader, loss_function, DEVICE)
