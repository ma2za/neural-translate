import torch
from datasets import load_dataset, Dataset
from evaluate import evaluator
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import get_constant_schedule_with_warmup


def load_opus_dataset(src, tgt, tokenizer):
    dataset = load_dataset("opus_euconst", f"{src}-{tgt}")

    dataset = dataset.shuffle(seed=42)

    dataset["validation"] = Dataset.from_dict(dataset["train"][:int(len(dataset["train"]) / 10)],
                                              features=dataset["train"].features)

    dataset["train"] = Dataset.from_dict(dataset["train"][int(len(dataset["train"]) / 10):],
                                         features=dataset["train"].features)

    def tokenization(sample):
        model_inputs = tokenizer(sample["translation"]["en"], padding=True, truncation=True)

        labels = tokenizer(text_target=sample["translation"]["it"], padding=True, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = dataset.map(tokenization, batched=False, batch_size=None)

    return dataset


def predict(model, tokenizer, sentence):
    temp = tokenizer.encode(sentence, return_tensors="pt").to("cuda")

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


def train_model():
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base").to("cuda")

    dataset = load_opus_dataset("en", "it", tokenizer)

    config = {
        "num_train_epochs": 25,
        "lr": 5e-05
    }

    training_args = Seq2SeqTrainingArguments(output_dir="test_trainer",
                                             auto_find_batch_size=True,
                                             evaluation_strategy="epoch",
                                             num_train_epochs=config["num_train_epochs"],
                                             save_total_limit=2)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id
    )

    optimizer = AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.999), eps=1e-08)
    lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=2000)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        optimizers=[optimizer, lr_scheduler],
        data_collator=data_collator,
    )

    trainer.train()
