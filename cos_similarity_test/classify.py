import csv
import json

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from operator import itemgetter

OUTPUT_PATH = "similar_prompt.json"

class USE:
    def __init__(self):
        self.encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def compute_sim(self, clean_texts, adv_texts):
        clean_embeddings = self.encoder(clean_texts)
        adv_embeddings = self.encoder(adv_texts)
        cosine_sim = tf.reduce_mean(tf.reduce_sum(clean_embeddings * adv_embeddings, axis=1))

        return float(cosine_sim.numpy())

# Specify the path to your CSV file
csv_file_path = 'harmful_behaviors.csv'

# Initialize an empty list to store the values
target_list = []

# Read the CSV file and append the values to the list
with open(csv_file_path, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        target_list.append(row['goal'])

use = USE()
score_list = []
max_len = len(target_list)
print(f"max_len = {max_len}")

for i in range(max_len):
    compare_list = []
    print(f"processing {i}")
    for j in range(max_len):
        if (i == j):
            sim_score = 0
        else:
            sim_score = use.compute_sim([target_list[i]], [target_list[j]])
        compare_list.append({"id": j, "score": sim_score})
    compare_list = sorted(compare_list, key=itemgetter('score'),reverse=True)
    score_list.append(compare_list[:9])
    
avg_score = []

for i in range(max_len):
    sum = 0
    for j in range(9):
        sum += score_list[i][j]['score']
    print(sum/9)
    avg_score.append({"id":i,"avg":sum/9})
avg_score = sorted(avg_score, key=itemgetter('avg'), reverse=True)
print(f"avg======================\n{avg_score[:9]}")
best_id = [avg_score[i]['id'] for i in range(max_len)]
print(f"best=======================\n{best_id[:9]}")

count = 0
used = [False] * max_len
output_list = []
for i in range(max_len):
    id = best_id[i]
    any_used = False
    if used[id]:
        any_used = True
    for j in range(9):
        if (used[score_list[id][j]['id']]):
            any_used = True
            break
    if not any_used:
        output_list.append({"id": count, 'input': target_list[id]})
        count += 1
        used[id] = True
        for k in range(9):
            t_id = score_list[id][k]['id']
            output_list.append({"id": count, 'input': target_list[t_id]})
            count += 1
            used[t_id] = True
    
with open(OUTPUT_PATH, 'w', encoding='utf-8') as output_file:
    json.dump(output_list, output_file, ensure_ascii=False, indent=2)

# print(output_list)