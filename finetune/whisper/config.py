# -------------------------------- *utf-8 encoding* -----------------------------------
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    output_dir: str = "/home/infinity/Documents/fluency_ai/model/out.csv"
    csv_path: str = "out.csv"
    batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 1e-6
    warmup_steps: int = 800
    max_steps: int = 80000
    gradient_accumulation_steps: int = 2
    eval_steps: int = 2000
    save_steps: int = 2000
    logging_steps: int = 500
    generation_max_length: int = 300
    num_workers: int = 4
    fp16: bool = True
