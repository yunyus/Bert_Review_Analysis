import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configuration
MODEL_NAME = 'roberta-base'
MODEL_PATH = 'roberta_model.pth'
MAX_LEN = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_LABELS = 5  # Adjust based on the specific dataset

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)


def predict_single_review(review):
    """
    Predict the star rating for a single review.
    """
    inputs = tokenizer(review, return_tensors='pt',
                       padding=True, truncation=True, max_length=MAX_LEN)
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).item()
    return predictions + 1  # Assuming labels are 1-indexed


def predict_dataset(dataset_path):
    """
    Predict star ratings for all reviews in a dataset.
    """
    df = pd.read_csv(dataset_path)
    reviews = df['comment'].tolist()
    inputs = tokenizer(reviews, return_tensors='pt',
                       padding=True, truncation=True, max_length=MAX_LEN)
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    df['predictions'] = predictions + 1  # Assuming labels are 1-indexed
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict star ratings for reviews.")
    parser.add_argument("--review", type=str,
                        help="Single review text to predict its star rating.")
    parser.add_argument("--dataset", type=str,
                        help="Path to dataset CSV file to predict star ratings for all reviews.")

    args = parser.parse_args()

    if args.review:
        print("Predicted star rating:", predict_single_review(args.review))
    elif args.dataset:
        predictions_df = predict_dataset(args.dataset)
        print(predictions_df.head())
        # Save predictions to a new CSV file
        predictions_df.to_csv('predictions.csv', index=False)
    else:
        print("Please provide either a review text or a dataset path.")


# Usage
# python inference.py --review "This is an example review."
# python inference.py --dataset "data/test.csv"
