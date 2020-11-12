# Chapter 11 - neural networks that understand language

# Download reviews.txt and labels.txt from here: https://github.com/udacity/deep-learning/tree/master/sentiment-network

import sys
import numpy as np

np.random.seed(1)

f = open(
    "/Users/jonas/Desktop/Books/Deep Learning - Andrew W Trask/Data/sentiment-network/reviews.txt"
)
raw_reviews = f.readlines()
f.close()

f = open(
    "/Users/jonas/Desktop/Books/Deep Learning - Andrew W Trask/Data/sentiment-network/labels.txt"
)
raw_labels = f.readlines()
f.close()

tokens = [set(x.split()) for x in raw_reviews]

vocab = set()
for sentiment in tokens:
    for word in sentiment:
        if len(word) > 0:
            vocab.add(word)
vocab = list(vocab)

word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i

input_dataset = []
for sentiment in tokens:
    sentiment_indices = []
    for word in sentiment:
        try:
            sentiment_indices.append(word2index[word])
        except:
            ""
    input_dataset.append(list(set(sentiment_indices)))

target_dataset = []
for label in raw_labels:
    if label == "positive\n":
        target_dataset.append(1)
    else:
        target_dataset.append(0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


alpha, iterations = (0.01, 2)
hidden_size = 100

weights_0_1 = 0.2 * np.random.random((len(vocab), hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, 1)) - 0.1

correct, total = (0, 0)
for iteration in range(iterations):

    for i in range(len(input_dataset) - 1000):
        x, y = (input_dataset[i], target_dataset[i])
        layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))  # embed + sigmoid
        layer_2 = sigmoid(np.dot(layer_1, weights_1_2))  # linear + softmax

        layer_2_delta = layer_2 - y  # comparing prediction with the truth
        layer_1_delta = layer_2_delta.dot(weights_1_2.T)  # backpropagation

        weights_0_1[x] -= layer_1_delta * alpha
        weights_1_2 -= np.outer(layer_1, layer_2_delta) * alpha

        if np.abs(layer_2_delta) < 0.5:
            correct += 1
        total += 1

        if i % 10 == 9:
            progress = str(i / float(len(input_dataset)))
            sys.stdout.write(
                f"\rIteraion: {iteration} Progress: {progress[2:4]}.{progress[4:6]}% Training Accuracy: {correct/total}%"
            )

correct, total = (0, 0)
for i in range(len(input_dataset) - 1000, len(input_dataset)):

    x = input_dataset[i]
    y = target_dataset[i]

    layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))
    layer_2 = sigmoid(np.dot(layer_1, weights_1_2))

    if np.abs(layer_2 - y) < 0.5:
        correct += 1
    total += 1

print(f"Test Accuracy: {correct/total}%")

# Iteraion: 1
# Progress: 95.99%
# Training Accuracy: 0.867%
# Test Accuracy: 0.851%


# Getting similar distance using Euclidian distance

from collections import Counter
import math


def similar(target="beautiful"):
    target_index = word2index[target]
    scores = Counter()
    for word, index in word2index.items():
        raw_difference = weights_0_1[index] - (weights_0_1[target_index])
        squared_difference = raw_difference * raw_difference
        scores[word] = -math.sqrt(sum(squared_difference))

    return scores.most_common(10)


print(similar("incredible"))

similar("incredible")
# ('refreshing', -0.6920210912798765),
# ('solid', -0.7447300155200107),
# ('noir', -0.7575210919848602),
# ('favorite', -0.7666592666915357),
# ('superb', -0.7757180844625264),
# ('fantastic', -0.7806356618179291),
# ('expecting', -0.789123600900986),
# ('rare', -0.8044447602416427),
# ('today', -0.8057024912861714)


print(similar("boring"))
# ('mess', -0.7888840154461145),
# ('disappointing', -0.7926966199524643),
# ('worse', -0.7931024341568277),
# ('terrible', -0.8333819339463026),
# ('lacks', -0.8368333751670779),
# ('badly', -0.8395922471433357),
# ('laughable', -0.8488228311204028),
# ('fails', -0.8509465131234148),
# ('horrible', -0.8565525441573284)
