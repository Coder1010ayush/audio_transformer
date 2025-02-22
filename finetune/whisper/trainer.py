import os
import torch
import evaluate
import logging
from tqdm import tqdm
from configuration.whisper_config import TrainingConfig
from data_loading import create_data_loaders
from configuration import initialize_components, save_checkpoint, save_final_model
from configuration import unknown_tokens

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    logging.info("Initializing training process...")
    config = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize components
    model, processor, tokenizer, optimizer, scheduler, scaler = initialize_components(
        config, unknown_tokens, device
    )
    logging.info("Model successfully initialized.")

    # Prepare datasets and data loaders
    train_loader, eval_loader = create_data_loaders(
        csv_path=config.csv_path,
        processor=processor,
        tokenizer=tokenizer,
        config=config,
    )
    logging.info("Dataloaders created.")

    global_step = 0
    optimizer.zero_grad()

    while global_step < config.max_steps:
        logging.info(f"Starting training epoch at global step {global_step}.")
        global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            config,
            global_step,
        )

        if global_step % config.eval_steps == 0:
            logging.info("Evaluating model...")
            evaluate_model(
                model, eval_loader, processor, tokenizer, device, config, global_step
            )
            logging.info("Saving checkpoint...")
            save_checkpoint(model, processor, config.output_dir, global_step)

    logging.info("Training complete. Saving final model...")
    save_final_model(model, processor, config.output_dir)


def train_epoch(
    model, train_loader, optimizer, scheduler, scaler, device, config, global_step
):
    model.train()
    progress_bar = tqdm(train_loader, desc=f"Training step {global_step}", leave=False)

    for batch in progress_bar:
        if global_step >= config.max_steps:
            break

        inputs = batch["input_features"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast():
            outputs = model(input_features=inputs, labels=labels)
            loss = outputs.loss / config.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (global_step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        if global_step % config.logging_steps == 0:
            total_norm = torch.norm(
                torch.stack(
                    [
                        torch.norm(p.grad.detach(), 2)
                        for p in model.parameters()
                        if p.grad is not None
                    ]
                ),
                2,
            ).item()
            logging.info(
                f"Step {global_step} - Loss: {loss.item() * config.gradient_accumulation_steps:.4f} - Gradient Norm: {total_norm:.4f}"
            )
        global_step += 1
    return global_step


def evaluate_model(model, loader, processor, tokenizer, device, config, step):
    model.eval()
    all_preds, all_refs = [], []
    metric = evaluate.load("wer")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            inputs = batch["input_features"].to(device)
            labels = batch["labels"].cpu().numpy()

            generated_ids = model.generate(
                inputs, max_length=config.generation_max_length
            )
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            labels[labels == -100] = tokenizer.pad_token_id
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_preds.extend(preds)
            all_refs.extend(refs)

    wer = 100 * metric.compute(predictions=all_preds, references=all_refs)
    logging.info(f"Step {step} - WER: {wer:.2f}%")
    model.train()


if __name__ == "__main__":
    main()
