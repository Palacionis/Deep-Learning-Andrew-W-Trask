# Chapter 12 - neural networks that write like Shakespear

# Download reviews.txt and labels.txt from here: https://github.com/udacity/deep-learning/tree/master/sentiment-network

import numpy as np
from collections import Counter

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




alpha, iterations = (0.01, 2)
hidden_size = 100

weights_0_1 = 0.2 * np.random.random((len(vocab), hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, 1)) - 0.1


norms = np.sum(weights_0_1 * weights_0_1,axis=1)
norms.resize(norms.shape[0],1)
normed_weights = weights_0_1 * norms

def make_sent_vect(words):
    indices = list(map(lambda x:word2index[x],filter(lambda x:x in word2index,words)))
    return np.mean(normed_weights[indices],axis=0)

reviews2vectors = list()
for review in tokens: # tokenized reviews
    reviews2vectors.append(make_sent_vect(review))
reviews2vectors = np.array(reviews2vectors)

def most_similar_reviews(review):
    v = make_sent_vect(review)
    scores = Counter()
    for i,val in enumerate(reviews2vectors.dot(v)):
        scores[i] = val
    most_similar = list()
    
    for idx,score in scores.most_common(3):
        most_similar.append(raw_reviews[idx])
    return most_similar

most_similar_reviews(['boring','bad', 'awful'])


# the characters are unlikeable and the script is awful . it  s a waste of the talents of deneuve and auteuil.
# long  boring  blasphemous . never have i been so glad to see ending credits roll.
# this movie is terrible but it has some good effects.


# identity matrices that change nothing. Yet.
import numpy as np
a = np.array([1,2,3])
b = np.array([0.1,0.2,0.3])
c = np.array([-1,-0.5,0])
d = np.array([0,0,0])

identity = np.eye(3)
print(identity)

# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]


print(a.dot(identity))
print(b.dot(identity))
print(c.dot(identity))
print(d.dot(identity))

# [1. 2. 3.]
# [0.1 0.2 0.3]
# [-1.  -0.5  0. ]
# [0. 0. 0.]


this = np.array([2,4,6])
movie = np.array([10,10,10])
rocks = np.array([1,1,1])

print(this + movie + rocks)
print((this.dot(identity) + movie).dot(identity) + rocks)

# [13 15 17]
# [13. 15. 17.]


# Forward propagation in Python

def softmax(x_):
    x = np.atleast_2d(x_)
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)

word_vects = {}
word_vects['yankees'] = np.array([[0.,0.,0.]])
word_vects['bears'] = np.array([[0.,0.,0.]])
word_vects['braves'] = np.array([[0.,0.,0.]])
word_vects['red'] = np.array([[0.,0.,0.]])
word_vects['socks'] = np.array([[0.,0.,0.]])
word_vects['lose'] = np.array([[0.,0.,0.]])
word_vects['defeat'] = np.array([[0.,0.,0.]])
word_vects['beat'] = np.array([[0.,0.,0.]])
word_vects['tie'] = np.array([[0.,0.,0.]])

sent2output = np.random.rand(3,len(word_vects))

identity = np.eye(3)

layer_0 = word_vects['red']
layer_1 = layer_0.dot(identity) + word_vects['socks']
layer_2 = layer_1.dot(identity) + word_vects['defeat']

pred = softmax(layer_2.dot(sent2output))
print(pred)

# because its softmax, we get:
# [[0.11111111 0.11111111 0.11111111 0.11111111 0.11111111 0.11111111
#   0.11111111 0.11111111 0.11111111]]

y = np.array([1,0,0,0,0,0,0,0,0]) # target one-hot vector for "yankees"

# To backpropagate we use:
pred_delta = pred - y
layer_2_delta = pred_delta.dot(sent2output.T)
defeat_delta = layer_2_delta * 1 # can ignore the "1" like prev. chapter
layer_1_delta = layer_2_delta.dot(identity.T)
socks_delta = layer_1_delta * 1 # again... can ignore the "1"
layer_0_delta = layer_1_delta.dot(identity.T)
alpha = 0.01
word_vects['red'] -= layer_0_delta * alpha
word_vects['socks'] -= socks_delta * alpha
word_vects['defeat'] -= defeat_delta * alpha
identity -= np.outer(layer_0,layer_1_delta) * alpha
identity -= np.outer(layer_1,layer_2_delta) * alpha
sent2output -= np.outer(layer_2,pred_delta) * alpha

# Get the data using bash:

# wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-1.tar.gz
# tar -xvf tasks_1-20_v1-1.tar.gz

import sys,random,math
from collections import Counter
import numpy as np

f = open('tasksv11/en/qa1_single-supporting-fact_train.txt','r')
raw = f.readlines()
f.close()

tokens = list()
for line in raw[0:1000]:
    tokens.append(line.lower().replace("\t","").replace("\n","").split(" ")[1:])


print(tokens[0:3])

# [['mary', 'moved', 'to', 'the', 'bathroom.'], 
# ['john', 'went', 'to', 'the', 'hallway.'], 
# ['where', 'is', 'mary?', 'bathroom1']]

vocab = set()
for sent in tokens:
    for word in sent:
        vocab.add(word)

vocab = list(vocab)
# vocab = [''.join(j for j in i if j.isalpha()) for i in vocab]

word2index = {}
for i,word in enumerate(vocab):
    word2index[word]=i
    
def words2indices(sentence):
    idx = list()
    for word in sentence:
        idx.append(word2index[word])
    return idx

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


embed_size = 10
# word embeddings
embed = (np.random.rand(len(vocab),embed_size) - 0.5) * 0.1
# embedding -> embedding (initially the identity matrix)
recurrent = np.eye(embed_size)
# sentence embedding for empty sentence
start = np.zeros(embed_size)
# embedding -> output weights
decoder = (np.random.rand(embed_size, len(vocab)) - 0.5) * 0.1
# one hot lookups (for loss function)
one_hot = np.eye(len(vocab))

def predict(sent):
    
    layers = list()
    layer = {}
    layer['hidden'] = start
    layers.append(layer)

    loss = 0

    # forward propagate
    for target_i in range(len(sent)):

        layer = {}

        # try to predict the next term
        layer['pred'] = softmax(layers[-1]['hidden'].dot(decoder))

        loss += -np.log(layer['pred'][sent[target_i]])

        # generate the next hidden state
        layer['hidden'] = layers[-1]['hidden'].dot(recurrent) + embed[sent[target_i]]
        layers.append(layer)

    return layers, loss

# forward
for iter in range(30000):
    alpha = 0.001
    sent = words2indices(tokens[iter%len(tokens)][1:])
    layers,loss = predict(sent) 

    # backward
    for layer_idx in reversed(range(len(layers))):
        layer = layers[layer_idx]
        target = sent[layer_idx-1]

        if(layer_idx > 0):  # if not the first layer
            layer['output_delta'] = layer['pred'] - one_hot[target]
            new_hidden_delta = layer['output_delta'].dot(decoder.transpose())

            # if the last layer - don't pull from a later one becasue it doesn't exist
            if(layer_idx == len(layers)-1):
                layer['hidden_delta'] = new_hidden_delta
            else:
                layer['hidden_delta'] = new_hidden_delta + layers[layer_idx+1]['hidden_delta'].dot(recurrent.transpose())
        else: # if the first layer
            layer['hidden_delta'] = layers[layer_idx+1]['hidden_delta'].dot(recurrent.transpose())


# weight update with arbitrary length
# forward
for iter in range(30000):
    alpha = 0.001
    sent = words2indices(tokens[iter%len(tokens)][1:])

    layers,loss = predict(sent) 

    # back propagate
    for layer_idx in reversed(range(len(layers))):
        layer = layers[layer_idx]
        target = sent[layer_idx-1]

        if(layer_idx > 0):
            layer['output_delta'] = layer['pred'] - one_hot[target]
            new_hidden_delta = layer['output_delta'].dot(decoder.transpose())

            # if the last layer - don't pull from a 
            # later one becasue it doesn't exist
            if(layer_idx == len(layers)-1):
                layer['hidden_delta'] = new_hidden_delta
            else:
                layer['hidden_delta'] = new_hidden_delta + layers[layer_idx+1]['hidden_delta'].dot(recurrent.transpose())
        else:
            layer['hidden_delta'] = layers[layer_idx+1]['hidden_delta'].dot(recurrent.transpose())

    # update weights
    start -= layers[0]['hidden_delta'] * alpha / float(len(sent))
    for layer_idx,layer in enumerate(layers[1:]):
        
        decoder -= np.outer(layers[layer_idx]['hidden'], layer['output_delta']) * alpha / float(len(sent))
        
        embed_idx = sent[layer_idx]
        embed[embed_idx] -= layers[layer_idx]['hidden_delta'] * alpha / float(len(sent))
        recurrent -= np.outer(layers[layer_idx]['hidden'], layer['hidden_delta']) * alpha / float(len(sent))
        
    if(iter % 1000 == 0):
        print("Perplexity:" + str(np.exp(loss/len(sent))))

# In information theory, perplexity is a measurement of how well a probability distribution or 
# probability model predicts a sample. It may be used to compare probability models. A low perplexity indicates 
# the probability distribution is good at predicting the sample.

# Perplexity:82.08056067497806
# Perplexity:81.95261365468883
# ...
# Perplexity:4.145747321517714
# Perplexity:4.054485930712955


sent_index = 4

l,_ = predict(words2indices(tokens[sent_index]))

print(tokens[sent_index])

for i,each_layer in enumerate(l[1:-1]):
    input = tokens[sent_index][i]
    true = tokens[sent_index][i+1]
    pred = vocab[each_layer['pred'].argmax()]
    print("Prev Input:" + input + (' ' * (12 - len(input))) +
          "True:" + true + (" " * (15 - len(true))) + "Pred:" + pred)

# ['sandra', 'moved', 'to', 'the', 'garden.']
# Prev Input:sandra      True:moved          Pred:is
# Prev Input:moved       True:to             Pred:to
# Prev Input:to          True:the            Pred:the
# Prev Input:the         True:garden.        Pred:bedroom.