# Deep Learning by Andrew W. Trask


# Creating a simple neural network with one input

weight = 0.1


def neural_network(input_, weight):
    prediction = input_ * weight

    return prediction


number_of_toes = [8.5, 9.5, 10, 9]
input_ = number_of_toes[0]
pred = neural_network(input_, weight)
print(pred)
# 0.850


# Creating a simple neural network with multiple inputs
# Predicting if a baseball team will win a game given:
# average number of toes per player
# current games won as a percentage
# count of fans in millions

weights = [0.1, 0.2, 0]

toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]


def w_sum(a, b):  # w_sum stands for weighted sum
    assert len(a) == len(b)
    output = 0

    for i in range(len(a)):
        output += a[i] * b[i]

    return output


def neural_network_multiple_inputs(input_, weights):
    pred = w_sum(input_, weights)

    return pred


input_ = [toes[0], wlrec[0], nfans[0]]
pred = neural_network_multiple_inputs(input_, weights)
print(pred)
# 0.980

# Vectors multiplication is a cornerstone technique in deep learning

vec_a = [1, 2, 3]
vec_b = [0.1, 1, 10]


def elementwise_multiplaction(vec_a, vec_b):
    assert len(vec_a) == len(vec_b)
    res = []

    for i in range(len(vec_a)):
        res.append(vec_a[i] * vec_b[i])

    return res


elementwise_multiplaction(vec_a, vec_b)
# [0.1, 2, 30]


def elementwise_addition(vec_a, vec_b):
    assert len(vec_a) == len(vec_b)
    res = []

    for i in range(len(vec_a)):
        res.append(vec_a[i] + vec_b[i])

    return res


elementwise_addition(vec_a, vec_b)
# [1.1, 3, 13]


def vector_sum(vec_a):
    return sum(vec_a)


vector_sum(vec_a)
# 6


def vector_average(vec_a):
    return sum(vec_a) / len(vec_a)


vector_average(vec_a)
# 2.0


# The intuition behind the dot product of a vector is just how similar they are, the less overlapping weights there are the more distinct the vectors are, for example:

a = [0, 1, 0, 1]
b = [1, 0, 1, 0]
c = [0, 1, 1, 0]
d = [0.5, 0, 0.5, 0]
e = [0, 1, -1, 0]

w_sum(a, b)
# returns 0 because they are not similar at all, 0*1 = 0 in all 4 items

w_sum(b, c)
# returns 1

w_sum(b, d)
# returns 1

w_sum(c, c)
# returns 2

w_sum(d, d)
# returns 0.5

w_sum(c, e)
# returns 0

# Refactoring code using NumPy
# Notice how we don't need to create w_sum() function
import numpy as np

weights = np.array([0.1, 0.2, 0])


def neural_network(input_, weights):
    pred = input_.dot(weights)
    return pred


toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65, 0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])

input_ = np.array([toes[0], wlrec[0], nfans[0]])
pred = neural_network(input_, weights)
print(pred)
# 0.980


# Making a prediction with multiple outputs
# Predicting if the players are happy or sad after a game

weights = [0.3, 0.2, 0.9]
wlrec = [0.65, 0.8, 0.8, 0.9]
input_ = wlrec[0]


def elementwise_multiplaction_2(number, vector):
    output = [0, 0, 0]
    assert len(output) == len(vector)

    for i in range(len(vector)):
        output[i] = number * vector[i]

    return output


def neural_network_with_multiple_outputs(input_, weights):
    pred = elementwise_multiplaction_2(input_, weights)
    return pred


pred = neural_network_with_multiple_outputs(input_, weights)
print(pred)
# [0.195, 0.130, 0.585]
# hurt     win    sad   predictions


# Predicting with multiple inputs and outputs

# toes   % win  # fans
weights = [[0.1, 0.1, -0.3], [0.1, 0.2, 0.0], [0.0, 1.3, 0.1]]  # hurt?  # win?  # sad?

toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

input_ = [toes[0], wlrec[0], nfans[0]]


def vect_mat_mul(vector, matrix):  # vector_matrix_multiplication
    assert len(vector) == len(matrix)
    output = [0, 0, 0]

    for i in range(len(vector)):
        output[i] = w_sum(vector, matrix[i])

    return output


def neural_network(input_, weights):
    pred = vect_mat_mul(input_, weights)
    return pred


pred = neural_network(input_, weights)
print(pred)
# [0.555, 0.98, 0.965]
# hurt?  win?  sad?


#   # toes      # win         # fans
# (8.5*0.1) + (0.65*0.1) + (1.2*(-0.3)) = 0.555 = hurt prediction
# (8.5*0.1) + (0.65*0.2) + (1.2*0.0)    = 0.980 = win prediction
# (8.5*0.0) + (0.65*1.3) + (1.2*0.1     = 0.965 = sad prediction


# Predicting on predictions / stacking neural networks

# toes   % win  # fans
ih_wgt = [
    [0.1, 0.2, -0.1],  # hid[0]
    [-0.1, 0.1, 0.9],  # hid[1]
    [0.1, 0.4, 0.1],
]  # hid[2]


# hid[0]  hid[1]  hid[2]
hp_wgt = [[0.3, 1.1, -0.3], [0.1, 0.2, 0.0], [0.0, 1.3, 0.1]]  # hurt?  # win?  # sad?

weights = [ih_wgt, hp_wgt]


def neural_network(input_, weights):
    hidden = vect_mat_mul(input_, weights[0])  # predicting the hidden layer
    # hidden =  [0.860, 0.294, 1.23]
    pred = vect_mat_mul(hidden, weights[1])  # predicting the output layer
    # pred = [0.213, 0.145, 0.506]
    return pred


toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

input_ = [toes[0], wlrec[0], nfans[0]]
pred = neural_network(input_, weights)
print(pred)
# [0.213, 0.145, 0.506]


# Using NumPy
import numpy as np

ih_wgt = np.array(
    [[0.1, 0.2, -0.1], [-0.1, 0.1, 0.9], [0.1, 0.4, 0.1]]  # hid[0]  # hid[1]
).T  # hid[2]


# hid[0]  hid[1]  hid[2]
hp_wgt = np.array(
    [[0.3, 1.1, -0.3], [0.1, 0.2, 0.0], [0.0, 1.3, 0.1]]  # hurt?  # win?
).T  # sad?

weights = [ih_wgt, hp_wgt]


def neural_network(input_, weights):
    hidden = input_.dot(weights[0])  # predicting the hidden layer
    print(hidden)
    pred = hidden.dot(weights[1])  # predicting the output layer
    print(pred)
    return pred


toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65, 0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])

input_ = np.array([toes[0], wlrec[0], nfans[0]])

pred = neural_network(input_, weights)
print(pred)
# [0.2135 0.145  0.5065]


# A quick NumPy primer

import numpy as np

a = np.array([0, 1, 2, 3])  # a vector
b = np.array([4, 5, 6, 7])  # another vector
c = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])  # a matrix

d = np.zeros((2, 4))  # (2x4 matrix of zeros)
e = np.random.rand(2, 5)  # random 2x5
# matrix with all numbers between 0 and 1

print(a)
# [0 1 2 3]
print(b)
# [4 5 6 7]
print(c)
# [[0 1 2 3]
#  [4 5 6 7]]
print(d)
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]]
print(e)
# [[0.06091447 0.46306551 0.66298891 0.75326352 0.79892192]
#  [0.52105573 0.45110005 0.86571316 0.05603345 0.78532425]]

print(a * 0.1)  # multiplies every number in vector "a" by 0.1
# [0.  0.1 0.2 0.3]
print(c * 0.2)  # multiplies every number in matrix "c" by 0.2
# [[0.  0.2 0.4 0.6]
#  [0.8 1.  1.2 1.4]]
print(a * b)  # multiplies elementwise between a and b (columns paired up)
# [ 0  5 12 21]
print(a * b * 0.2)  # elementwise multiplication then multiplied by 0.2
# [0.  1.  2.4 4.2]
print(a * c)  # since c has the same number of columns as a, this performs
# elementwise multiplication on every row of the matrix "c"
# [[ 0  1  4  9]
#  [ 0  5 12 21]]
print(a * e)  # since a and e don't have the same number of columns, this
# throws a ValueError: operands could not be broadcast together with shapes (4,) (2,5)


a = np.zeros((1, 4))  # vector of length 4
b = np.zeros((4, 3))  # matrix with 4 rows & 3 columns

c = a.dot(b)
print(c.shape)
# (1, 3)

a = np.zeros((2, 4))  # matrix with 2 rows and 4 columns
b = np.zeros((4, 3))  # matrix with 4 rows & 3 columns

c = a.dot(b)
print(c.shape)  # outputs (2,3)

e = np.zeros((2, 1))  # matrix with 2 rows and 1 columns
f = np.zeros((1, 3))  # matrix with 1 row & 3 columns

g = e.dot(f)
print(g.shape)  # outputs (2,3)

h = np.zeros((5, 4)).T  # matrix with 4 rows and 5 columns
i = np.zeros((5, 6))  # matrix with 6 rows & 5 columns

j = h.dot(i)
print(j.shape)  # outputs (4,6)

h = np.zeros((5, 4))  # matrix with 5 rows and 4 columns
i = np.zeros((5, 6))  # matrix with 5 rows & 6 columns
j = h.dot(i)
print(j.shape)  # throws an error

