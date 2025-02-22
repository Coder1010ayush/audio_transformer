# -------------------------------- *utf-8 encoding* -----------------------------------
import torch
import os
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AdamW,
    get_cosine_schedule_with_warmup,
)


def initialize_components(config, new_tokens, device):
    # Initialize processor and model
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-small", language="en", task="transcribe"
    )
    tokenizer.add_tokens(list(new_tokens))

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(
        device
    )
    model.resize_token_embeddings(len(tokenizer))
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)

    return model, processor, tokenizer, optimizer, scheduler, scaler


def save_checkpoint(model, processor, tokenizer, output_dir, step):
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"Saved checkpoint to {checkpoint_dir}")


def save_final_model(model, processor, tokenizer, output_dir):
    final_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final model to {final_dir}")


unknown_tokens = [
    "ʂ",
    "dʰ",
    "l",
    "d",
    "õ",
    "ɑ",
    "s",
    "ɟ",
    "eə",
    "ĩ",
    "o",
    "b",
    "ɜ",
    "ə",
    "u",
    "i",
    "aʊ",
    "aɪ",
    "a",
    "r",
    "ɖʰ",
    "ɲ",
    "tʰ",
    "ã",
    "w",
    "ɹ",
    "m",
    "p",
    "ẽ",
    "ɖ",
    "θ",
    "ɛ",
    "aɪə",
    "ʌ",
    "ŋ",
    "z",
    "eɪ",
    "x",
    "ɟʰ",
    "ɔ̃",
    "sh",
    "ʒ",
    "ɪ",
    "ʊə",
    "c",
    "f",
    "ɔ",
    "n",
    "ɔɪ",
    "kʰ",
    "ɐ",
    "ʋ",
    "ɾ",
    "e",
    "ɳ",
    "ɒ",
    "ʊ",
    "ɡʰ",
    "h",
    "t",
    "ʃ",
    "əʊ",
    "ɡ",
    "ʈʰ",
    "əl",
    "cʰ",
    "bʰ",
    "iə",
    "v",
    "k",
    "g",
    "ʈ",
    "j",
    "pʰ",
]
