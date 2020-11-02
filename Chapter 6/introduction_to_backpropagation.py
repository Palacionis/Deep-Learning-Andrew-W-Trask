import numpy as np

weights = np.array([0.5, 0.48, -0.7])
alpha = 0.1

streetlights = np.array(
    [[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1]]
)

walk_vs_stop = np.array([0, 1, 0, 1, 1, 0])

input_ = streetlights[0]  # first streetlight
goal_prediction = walk_vs_stop[0]  # first light combinations is stop

for iteration in range(20):
    prediction = input_.dot(weights)
    error = (prediction - goal_prediction) ** 2
    delta = prediction - goal_prediction
    weights -= alpha * (input_ * delta)

    print(f"Error: {error}\nPrediction: {prediction}")

# Error: 0.03999999999999998
# Prediction: -0.19999999999999996
# ...
# Error: 8.307674973656916e-06
# Prediction: -0.002882303761517324


# -------------------------------------------
# So far all of the neural networks only handeled a single input, how to train it on all dataset?


weights = np.array([0.5, 0.48, -0.7])
alpha = 0.1

streetlights = np.array(
    [[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1]]
)

walk_vs_stop = np.array([0, 1, 0, 1, 1, 0])

input_ = streetlights[0]  # first streetlight
goal_prediction = walk_vs_stop[0]  # first light combinations is stop


for iteration in range(40):
    error_for_all_lights = 0
    for row_index in range(len(walk_vs_stop)):
        input_ = streetlights[row_index]
        goal_prediction = walk_vs_stop[row_index]

        prediction = input_.dot(weights)

        error = (goal_prediction - prediction) ** 2
        error_for_all_lights += error

        delta = prediction - goal_prediction
        weights -= alpha * (input_ * delta)
        print(f"Prediction: {prediction}")
    print(f"Error: {error_for_all_lights}\n")


# -------------------------------------------
# Adding backpropagation in code

import numpy as np

np.random.seed(1)

streetlights = np.array(
    [[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1]]
)

walk_vs_stop = np.array([0, 1, 0, 1, 1, 0])


def relu(x):
    return (x > 0) * x


def relu2deriv(output):
    return output > 0


alpha = 0.2
hidden_layer_size = 4

# random weights from the first layer to the second
weights_0_1 = 2 * np.random.random((3, hidden_layer_size)) - 1
# random weights from the second layer to the output
weights_1_2 = 2 * np.random.random((hidden_layer_size, 1)) - 1

for iteration in range(60):
    layer_2_error = 0
    for i in range(len(streetlights)):
        layer_0 = streetlights[i : i + 1]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = relu(np.dot(layer_1, weights_1_2))

        layer_2_error += np.sum(layer_2 - walk_vs_stop[i : i + 1]) ** 2

        layer_2_delta = walk_vs_stop[i : i + 1] - layer_2
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)

        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

    if iteration % 9 == 0:
        print(f"Error: {layer_2_error}")


# Error: 3.1120884967431595
# Error: 0.31298811341557736
# Error: 0.0540487565336386
# Error: 0.005919784786873184
# Error: 0.00047166834729714375
# Error: 3.338825575496048e-05
# Error: 2.2837145520392978e-06
