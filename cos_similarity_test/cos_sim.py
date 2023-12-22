import csv

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


class USE:
    def __init__(self):
        self.encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def compute_sim(self, clean_texts, adv_texts):
        clean_embeddings = self.encoder(clean_texts)
        adv_embeddings = self.encoder(adv_texts)
        cosine_sim = tf.reduce_mean(tf.reduce_sum(clean_embeddings * adv_embeddings, axis=1))

        return float(cosine_sim.numpy())

# Specify the path to your CSV file
csv_file_path = 'harmful_strings.csv'

# Initialize an empty list to store the values
target_list = []

# Read the CSV file and append the values to the list
with open(csv_file_path, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        target_list.append(row['target'])

use = USE()
score_list = []

score_list.append(use.compute_sim([target_list[0]], [target_list[1]]))
score_list.append(use.compute_sim([target_list[0]], [target_list[2]]))
print(score_list)