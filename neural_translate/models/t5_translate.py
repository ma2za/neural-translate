import random
from functools import partial

import numpy as np
import torch
from datasets import load_dataset, Dataset
from evaluate import evaluator
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq
from transformers import T5TokenizerFast, T5ForConditionalGeneration

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_opus_dataset(src, tgt, tokenizer):
    dataset = load_dataset("opus_euconst", f"{src}-{tgt}")

    dataset = dataset.shuffle(seed=42)

    dataset["validation"] = Dataset.from_dict(dataset["train"][:int(len(dataset["train"]) / 10)],
                                              features=dataset["train"].features)

    dataset["train"] = Dataset.from_dict(dataset["train"][int(len(dataset["train"]) / 10):],
                                         features=dataset["train"].features)

    def tokenization(sample, truncation=False, max_length=None):
        model_inputs = tokenizer(sample["translation"]["en"],
                                 padding=False,
                                 truncation=truncation,
                                 max_length=max_length)

        labels = tokenizer(text_target=sample["translation"]["it"],
                           padding=False,
                           truncation=truncation,
                           max_length=max_length)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset["train"] = dataset["train"].map(partial(tokenization, truncation=True,
                                                    max_length=100), batched=False,
                                            batch_size=None,
                                            remove_columns=["translation"])
    dataset["validation"] = dataset["validation"].map(partial(tokenization, truncation=True,
                                                              max_length=512),
                                                      batched=False, batch_size=None,
                                                      remove_columns=["translation"])

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
    "lr": 1e-05,
    "epochs": 50,
    "batch_size": 4,
    "warmup_ratio": 0.2
}

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    padding=True
)

train_loader = DataLoader(
    dataset["train"],
    batch_size=config["batch_size"],
    collate_fn=data_collator,
    drop_last=False,
    num_workers=0,
    shuffle=True,
    pin_memory=True
)

valid_loader = DataLoader(
    dataset["validation"],
    batch_size=4,
    collate_fn=data_collator,
    drop_last=False,
    num_workers=0,
    pin_memory=True
)

optimizer = AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)

num_training_steps = float(len(train_loader) * config["epochs"])

num_warmup_steps = num_training_steps * config["warmup_ratio"]


def lr_lambda(x: float, warmup: float, total: float):
    return (x + 1) / warmup if x < warmup else (total - x) / (total - warmup)


lr_scheduler = LambdaLR(optimizer, partial(lr_lambda, warmup=num_warmup_steps,
                                           total=num_training_steps))


def evaluation(model, dataloader):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**inputs)
            total_samples += len(batch["labels"])
            total_loss += outputs.loss.detach().cpu() * len(batch["labels"])
    return total_loss / total_samples


def train_epoch(model, optimizer, lr_scheduler, train_loader):
    model.train()
    for step, inputs in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = model(**inputs)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        lr_scheduler.step()

        model.zero_grad()
    return outputs.loss.detach().cpu().item(), lr_scheduler.get_last_lr()


best_loss = 100
model.zero_grad()
for epoch in range(config["epochs"]):
    train_loss, last_lr = train_epoch(model, optimizer, lr_scheduler, train_loader)
    eval_loss = evaluation(model, valid_loader)

    if eval_loss < best_loss:
        best_loss = eval_loss
        torch.save(model.state_dict(), "pytorch_model.bin")

    log_dict = {"eval/loss": eval_loss.item(), "train/loss": train_loss}
    print(f"EPOCH: {epoch}, {log_dict}")

model_state = torch.load("pytorch_model.bin")
model.load_state_dict(model_state)
