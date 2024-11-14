# -*- coding: utf-8 -*-

import json
import time

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

finetuned_model_path = "/opt/qs/models/xlm-roberta-base-finetune"
with open(finetuned_model_path + '/label_to_id.json', 'r') as f:
    label_to_id = json.load(f)
id_to_label = {int(v): k for k, v in label_to_id.items()}

finetunedM = AutoModelForSequenceClassification.from_pretrained(finetuned_model_path, device_map={"": "cuda:1"})
tokenizerM = AutoTokenizer.from_pretrained(finetuned_model_path)

df_test = pd.read_excel('data/generate_testing_questions_result.xlsx', engine='openpyxl')
sequences = df_test["test_question"].values

pred_labels = []
scores_ = []
TFs = []
costs = []
for i, text in enumerate(sequences):
    start = time.time()
    tokens = tokenizerM([text], padding="max_length", truncation=True, return_tensors="pt").to("cuda:1")
    outputs = finetunedM(**tokens)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    scores, predicted_labels = torch.max(predictions, dim=-1)

    pred_labels.append(id_to_label[predicted_labels.tolist()[0]])
    scores_.append(scores.tolist()[0])
    costs.append(time.time() - start)
    # print("{} cost: {}".format(i, time.time() - start))

df_test["pred_label"] = pred_labels
df_test["score"] = scores_
df_test["cost"] = costs
df_test["TF"] = df_test["pred_label"] == df_test["topic"]
df_test.to_csv("data/test_result.csv", encoding="utf-8", index=False)
