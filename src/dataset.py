import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class ReviewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        review = str(self.df.loc[idx, 'comment'])  # Ensure review is string
        label = int(self.df.loc[idx, 'stars']) - 1

        encoded = self.tokenizer(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            truncation=True
        )

        input_ids = encoded['input_ids']
        attn_mask = encoded['attention_mask']

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attn_mask': torch.tensor(attn_mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


def create_data_loaders(train_df, val_df, test_df, tokenizer, max_length=256, batch_size=32, num_workers=4):
    train_dataset = ReviewsDataset(train_df, tokenizer, max_length)
    val_dataset = ReviewsDataset(val_df, tokenizer, max_length)
    test_dataset = ReviewsDataset(test_df, tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
