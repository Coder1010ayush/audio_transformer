# -------------------------------- *utf-8 encoding* -----------------------------------
import torch
import pandas as pd
import librosa
import os
from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    def __init__(self, csv_path, processor, tokenizer, max_label_length=300):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_label_length = max_label_length
        self._preprocess_data()

    def _preprocess_data(self):
        self.df = self.df[self.df["label"].str.len() <= self.max_label_length]
        # self.df = self.df.dropna(subset=["audio_path", "label"])
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _load_audio(self, audio_path):
        # Load and resample audio to 16kHz
        audio, sr = librosa.load(
            os.path.join("/home/infinity/Documents/fluency_ai/data/audio", audio_path),
            sr=16000,
        )
        return audio

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            audio_path = row["audio_path"]
            label = row["label"]

            audio = self._load_audio(audio_path)
            input_features = self.processor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_features[0]

            labels = self.tokenizer(
                label,
                max_length=self.max_label_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids[0]

            return {"input_features": input_features, "labels": labels}

        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None


class DataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        input_features = [{"input_features": b["input_features"]} for b in batch]
        processed_batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": b["labels"]} for b in batch]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        processed_batch["labels"] = labels
        return processed_batch


def create_data_loaders(csv_path, processor, tokenizer, config):
    full_dataset = AudioDataset(csv_path, processor, tokenizer)

    train_size = int(0.99 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=DataCollator(processor),
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        collate_fn=DataCollator(processor),
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
