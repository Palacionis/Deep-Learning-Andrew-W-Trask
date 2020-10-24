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


def w_sum(a, b):
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
