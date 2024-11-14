# -*- coding: utf-8 -*-

import shutil
import numpy as np
from datasets import load_metric
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
import datasets
import json

model_name = "/opt/qs/models/xlm-roberta-base"
finetuned_model_path = "/opt/qs/models/xlm-roberta-base-finetune"

# 1. 加载数据集和tokenizer
dataset = datasets.load_dataset('csv', data_files={'train': 'data/train_data.csv', 'test': 'data/test_data.csv'})
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True, model_max_length=512)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 2. 加载模型
label_to_id_file = "model/label_to_id.json"
with open(label_to_id_file, 'r') as f:
    label_to_id = json.load(f)
id_to_label = {int(v): k for k, v in label_to_id.items()}
# 这里指定device_map={"": "cuda:1"}无效，使用CUDA_VISIBLE_DEVICES=1指定
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_to_id), label2id=label_to_id, id2label=id_to_label, trust_remote_code=True)

# 3. 模型训练
training_args = TrainingArguments(
    output_dir='model',
    evaluation_strategy='steps',
    eval_steps=10,
    logging_steps=100,  # 打印train loss等信息，较耗时
    save_total_limit=5,
    num_train_epochs=40,
    save_steps=20,
    load_best_model_at_end=True,
    per_device_train_batch_size=64,
    # learning_rate=1e-4
)
metric = load_metric("metrics/accuracy.py")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics
)
trainer.train()
trainer.evaluate()  # 直接跑传入的验证集

# 保存模型
tokenizer.save_pretrained(finetuned_model_path)
trainer.save_model(finetuned_model_path)
shutil.copy2(label_to_id_file, finetuned_model_path)
print("train finished...")
