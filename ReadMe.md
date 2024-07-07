# Review Sentiment Analysis

This project provides a framework for sentiment analysis of review comments using BERT. The project includes data loading, preprocessing, model training, and inference functionalities. It predicts the star number of a comment

## Project Structure

```
review_sentiment_analysis/
│
├── data/
│   └── reviews_comments_stars.csv
├── src/
│   ├── config.py
│   ├── data_preparation.py
│   ├── dataset.py
│   ├── inference.py
│   ├── model_training.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository:

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

3. Place your dataset in the `data` directory. The dataset should be a CSV file named `reviews_comments_stars.csv`.

## Usage

### Training

To train the model, run:

```
python src/train.py
```

### Inference

To predict star ratings for a single review or an entire dataset, run:

```
python src/inference.py --review "This is an example review."
```

or

```
python src/inference.py --dataset "data/reviews_comments_stars.csv"
```
