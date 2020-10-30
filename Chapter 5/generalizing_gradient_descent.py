# Chapter 5 - generalizing gradient descent

# multiple weights at a time


def neural_network(input_, weights):
    output = 0
    for i in range(len(input_)):
        output += input_[i] * weights[i]
    return round(output, 2)


def ele_mul(scalar, vector):
    output = []
    for i in range(len(vector)):
        output.append(round(vector[i] * scalar, 2))
    return output


toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

win_or_lose_binary = [1, 1, 0, 1]
true = win_or_lose_binary[0]  # win

alpha = 0.01
weights = [0.1, 0.2, -0.1]

input_ = [toes[0], wlrec[0], nfans[0]]

for x in range(3):
    pred = neural_network(input_, weights)
    error = (pred - true) ** 2
    delta = pred - true

    weight_deltas = ele_mul(delta, input_)

    print(f"Iteration: {round(x+1,2)}")
    print(f"Prediction: {round(pred,2)}")
    print(f"Error: {round(error,2)}")
    print(f"Delta: {round(delta,2)}")
    print(f"Weights: {weights}")
    print(f"Weight Deltas {weight_deltas}")

    for i in range(len(weights)):
        weights[i] -= round(alpha * weight_deltas[i], 2)


# Iteration: 1
# Prediction: 0.86
# Error: 0.02
# Delta: -0.14
# Weights: [0.1, 0.2, -0.1]
# Weight Deltas [-1.19, -0.09, -0.17]
# Iteration: 2
# Prediction: 0.09
# Error: 0.83
# Delta: -0.91
# Weights: [0.01, 0.0, 0.0]
# Weight Deltas [-7.74, -0.59, -1.09]
# Iteration: 3
# Prediction: 0.7
# Error: 0.09
# Delta: -0.3
# Weights: [0.08, 0.01, 0.01]
# Weight Deltas [-2.55, -0.2, -0.36]


# -----------------------------------------
# Freezing one weight: what does it do?
# Even after freezing the first weight, error still went down to 0, because the error is shared, when one weight finds an error of 0, all weights do.


def neural_network(input_, weights):
    output = 0
    for i in range(len(input_)):
        output += input_[i] * weights[i]
    return round(output, 2)


def ele_mul(scalar, vector):
    output = []
    for i in range(len(vector)):
        output.append(round(vector[i] * scalar, 2))
    return output


toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

win_or_lose_binary = [1, 1, 0, 1]
true = win_or_lose_binary[0]  # win

alpha = 0.3  # changed
weights = [0.1, 0.2, -0.1]

input_ = [toes[0], wlrec[0], nfans[0]]

for x in range(3):
    pred = neural_network(input_, weights)
    error = (pred - true) ** 2
    delta = pred - true

    weight_deltas = ele_mul(delta, input_)
    weight_deltas[0] = 0  # setting the first weight, to always be 0

    print(f"Iteration: {round(x+1,2)}")
    print(f"Prediction: {round(pred,2)}")
    print(f"Error: {round(error,2)}")
    print(f"Delta: {round(delta,2)}")
    print(f"Weights: {weights}")
    print(f"Weight Deltas {weight_deltas}")

    for i in range(len(weights)):
        weights[i] -= round(alpha * weight_deltas[i], 2)

# Gradient escent learning with multiple outputs

# Instead of predicting just whether the team won or lost,
# now we're also predicting whether they are happy/sad AND the
# percentage of the team that is hurt. We are making this prediction using only
# the current win/loss record.

weights = [0.3, 0.2, 0.9]


def neural_network(input_, weights):
    pred = ele_mul(input_, weights)
    return pred


def scalar_ele_mul(number, vector):
    output = [0, 0, 0]
    assert len(output) == len(vector)
    for i in range(len(vector)):
        output[i] = number * vector[i]

    return output


wlrec = [0.65, 1.0, 1.0, 0.9]

hurt = [0.1, 0.0, 0.0, 0.1]
win = [1, 1, 0, 1]
sad = [0.1, 0.0, 0.1, 0.2]

input_ = wlrec[0]
true = [hurt[0], win[0], sad[0]]

pred = neural_network(input_, weights)

error = [0, 0, 0]
delta = [0, 0, 0]

for i in range(len(true)):
    error[i] = (pred[i] - true[i]) ** 2
    delta[i] = pred[i] - true[i]


weight_deltas = scalar_ele_mul(input_, delta)

alpha = 0.1

for i in range(len(weights)):
    weights[i] -= weight_deltas[i] * alpha

print(f"Weights: {weights}")
print(f"Weight Deltas: {weight_deltas}")


# Weights: [0.293825, 0.25655, 0.868475]
# Weight Deltas: [0.061, -0.5655, 0.315]