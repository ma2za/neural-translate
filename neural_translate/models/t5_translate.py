import random
from functools import partial

import numpy as np
import torch
import wandb
from datasets import load_dataset, Dataset
from evaluate import evaluator
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import T5TokenizerFast, T5ForConditionalGeneration

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Commented out IPython magic to ensure Python compatibility.
# %env WANDB_PROJECT=t5_translate_en_it

wandb.login()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_opus_dataset(src, tgt, tokenizer):
    dataset = load_dataset("opus_euconst", f"{src}-{tgt}")

    dataset = dataset.shuffle(seed=42)

    dataset["validation"] = Dataset.from_dict(dataset["train"][:int(len(dataset["train"]) / 10)],
                                              features=dataset["train"].features)

    dataset["train"] = Dataset.from_dict(dataset["train"][int(len(dataset["train"]) / 10):],
                                         features=dataset["train"].features)

    def tokenization(sample):
        # TODO do not truncate validation
        # TODO add truncated tokens as new samples
        model_inputs = tokenizer(sample["translation"]["en"], padding=True,
                                 truncation=True, max_length=100)

        labels = tokenizer(text_target=sample["translation"]["it"], padding=True,
                           truncation=True, max_length=100)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = dataset.map(tokenization, batched=False, batch_size=None, remove_columns=["translation"])

    return dataset


def predict(model, tokenizer, sentence):
    temp = tokenizer.encode(sentence, return_tensors="pt").to(DEVICE)

    model.eval()
    with torch.no_grad():
        out = model.generate(temp)

    return tokenizer.decode(out[0], skip_special_tokens=True)


def evaluate(model, tokenizer):
    with open("../data/dataset/newssyscomb2009.en", "r") as file:
        data_en = file.read()

    with open("../data/dataset/newssyscomb2009.it", "r") as file:
        data_it = file.read()

    test_dataset = Dataset.from_dict({"text": data_en.split("\n"), "label": data_it.split("\n")})

    task_evaluator = evaluator("translation")

    results = task_evaluator.compute(
        model_or_pipeline=model,
        data=test_dataset,
        tokenizer=tokenizer,
        metric="bleu")
    return results


tokenizer = T5TokenizerFast.from_pretrained("t5-base")

model = T5ForConditionalGeneration.from_pretrained("t5-base").to(DEVICE)

dataset = load_opus_dataset("en", "it", tokenizer)

config = {
    "lr": 5e-05,
    "epochs": 25,
    "batch_size": 32,
    "warmup_ratio": 0.2
}

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=tokenizer.pad_token_id
)

train_loader = DataLoader(
    dataset["train"],
    batch_size=config["batch_size"],
    collate_fn=data_collator,
    drop_last=False,
    num_workers=0,
    pin_memory=True
)

optimizer = AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.999), eps=1e-08)

num_training_steps = float(len(train_loader) * config["epochs"])

num_warmup_steps = num_training_steps * config["warmup_ratio"]


def lr_lambda(x: float, warmup: float, total: float):
    return (x + 1) / warmup if x < warmup else (total - x) / (total - warmup)


lr_scheduler = LambdaLR(optimizer, partial(lr_lambda, warmup=num_warmup_steps,
                                           total=num_training_steps))

scaler = GradScaler()

trainer = Seq2SeqTrainer(
    model=model,
    args=Seq2SeqTrainingArguments(output_dir="dummy_dir"),
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
)


def train_epoch(model, optimizer, scaler, lr_scheduler, train_loader):
    model.train()
    for step, inputs in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(**inputs)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        scaler.scale(outputs.loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        model.zero_grad()
    return outputs.loss.detach().cpu().item(), lr_scheduler.get_last_lr()


wandb.init(project="t5_translate_en_it", config=config)

model.zero_grad()
for epoch in range(config["epochs"]):
    train_loss, last_lr = train_epoch(model, optimizer, scaler, lr_scheduler, train_loader)
    eval_results = trainer.evaluate()
    log_dict = {"eval/loss": eval_results['eval_loss'],
                "train/loss": train_loss}
    print(log_dict)
    wandb.log(log_dict)
wandb.finish()
