# -*- coding: utf-8 -*-

import pandas as pd
import sklearn
import json

# 训练数据处理
df_train = pd.read_excel('data/Harmony.xlsx', engine='openpyxl', usecols=['Visitor Questions', 'Topic Name'])
df_train['Visitor Questions'] = df_train['Visitor Questions'].str.split('|')
df_train = df_train.explode('Visitor Questions').reset_index(drop=True)
df_train['Visitor Questions'] = df_train['Visitor Questions'].str.strip()
df_train = df_train[df_train['Visitor Questions'] != ""]
df_train.rename(columns={'Visitor Questions': 'text', 'Topic Name': 'label'}, inplace=True)
df_train = sklearn.utils.shuffle(df_train)
print(df_train)

# label设置
unique_labels = df_train['label'].unique()
label_to_id = {label: i for i, label in enumerate(unique_labels)}
df_train['label'] = df_train['label'].map(label_to_id)
df_train['label'] = df_train['label'].astype(int)
with open('model/label_to_id.json', 'w') as f:
    f.write(json.dumps(label_to_id))
print(df_train)
df_train.to_csv("data/train_data.csv", encoding="utf-8", index=False)

# 测试数据处理
df_test = pd.read_excel('data/generate_testing_questions_result.xlsx', engine='openpyxl', usecols=['test_question', 'topic'])
df_test.rename(columns={'test_question': 'text', 'topic': 'label'}, inplace=True)
df_test['label'] = df_test['label'].map(label_to_id)
df_test['label'] = df_test['label'].astype(int)
print(df_test)
df_test.to_csv("data/test_data.csv", encoding="utf-8", index=False)
