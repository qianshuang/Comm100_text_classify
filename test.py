# -*- coding: utf-8 -*-

import json

label_to_id_file = "model/label_to_id.json"
with open(label_to_id_file, 'r') as f:
    label_to_id = json.load(f)
print(label_to_id)
print(label_to_id["Talk to human agent"])
print(type(label_to_id["Talk to human agent"]))
