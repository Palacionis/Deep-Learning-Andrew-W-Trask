# Chapter 8 - learning signal and ignoring noise

# Predicting hand-written digits

import sys, numpy as np
from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
images, labels = (x_train[:1000].reshape(1000, 28 * 28) / 255, y_train[:1000])
one_hot_labels = np.zeros((len(labels), 10))

for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

test_images = x_test.reshape(len(x_test), 28 * 28) / 255
test_labels = np.zeros((len(y_test), 10))

for i, l in enumerate(y_test):
    test_labels[i][l] = 1

np.random.seed(1)
relu = lambda x: (x > 0) * x
relu2deriv = lambda x: x > 0

alpha = 0.005
iterations = 350
hidden_size = 40
pixels_per_image = 784  # 28*28
num_labels = 10

weights_0_1 = 0.2 * np.random.random((pixels_per_image, hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

for j in range(iterations):
    error = 0
    correct_count = 0

    for i in range(len(images)):
        layer_0 = images[[i]]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)

        error += np.sum((labels[i : i + 1] - layer_2)) ** 2
        correct_count += int(np.argmax(layer_2)) == np.argmax(labels[i : i + 1])

        layer_2_delta = labels[i : i + 1] - layer_2
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)

        weights_1_2 += alpha * (layer_1.T.dot(layer_2_delta))
        weights_0_1 += alpha * (layer_0.T.dot(layer_1_delta))

    sys.stdout.write(
        "\r"
        + "I:"
        + str(j)
        + "Error:"
        + str(error / float(len(images)))[:5]
        + "Correct"
        + str(correct_count / float(len(images)))
    )

# I:349
# Error:0.003
# Correct0.999

# --------------------------------------------------
# Checking neural network with test images and labels

if j % 10 == 0 or j == iterations - 1:
    error, correct_count = 0.0, 0
    for i in range(len(test_images)):

        layer_0 = test_images[i : i + 1]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)

        error += np.sum((test_labels[i : i + 1] - layer_2) ** 2)
        correct_count += int(np.argmax(layer_2) == np.argmax(test_labels[i : i + 1]))


sys.stdout.write(
    "Test-Error: "
    + str(error / float(len(test_images)))[:5]
    + " Test-Accuracy: "
    + str(correct_count / float(len(test_images)))
)

# Test-Error: 0.355
# Test-Accuracy: 0.829

# --------------------------------------------------
# Adding dropout to cancel the noise


(x_train, y_train), (x_test, y_test) = mnist.load_data()
images, labels = (x_train[:1000].reshape(1000, 28 * 28) / 255, y_train[:1000])
one_hot_labels = np.zeros((len(labels), 10))

for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

test_images = x_test.reshape(len(x_test), 28 * 28) / 255
test_labels = np.zeros((len(y_test), 10))

for i, l in enumerate(y_test):
    test_labels[i][l] = 1

np.random.seed(1)
relu = lambda x: (x > 0) * x
relu2deriv = lambda x: x > 0

alpha = 0.005
iterations = 300
hidden_size = 40
pixels_per_image = 784  # 28*28
num_labels = 10

weights_0_1 = 0.2 * np.random.random((pixels_per_image, hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

for j in range(iterations):
    error = 0
    correct_count = 0

    for i in range(len(images)):
        layer_0 = images[[i]]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1 *= dropout_mask * 2
        layer_2 = np.dot(layer_1, weights_1_2)

        error += np.sum((labels[i : i + 1] - layer_2)) ** 2
        correct_count += int(np.argmax(layer_2)) == np.argmax(labels[i : i + 1])

        layer_2_delta = labels[i : i + 1] - layer_2
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)
        layer_1_delta *= dropout_mask

        weights_1_2 += alpha * (layer_1.T.dot(layer_2_delta))
        weights_0_1 += alpha * (layer_0.T.dot(layer_1_delta))


if j % 10 == 0 or j == iterations - 1:
    test_error, test_correct_count = 0.0, 0
    for i in range(len(test_images)):

        layer_0 = test_images[i : i + 1]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)

        test_error += np.sum((test_labels[i : i + 1] - layer_2) ** 2)
        test_correct_count += int(
            np.argmax(layer_2) == np.argmax(test_labels[i : i + 1])
        )

        sys.stdout.write(
            "\r"
            + "I:"
            + str(j)
            + "\nTest-Error: "
            + str(test_error / float(len(test_images)))[:5]
            + "\nTest-Accuracy: "
            + str(test_correct_count / float(len(test_images)))[:5]
            + "\nTrain-Error: "
            + str(error / float(len(images)))[:5]
            + "\nTrain-Accuracy: "
            + str(correct_count / float(len(images)))
        )


# I:299
# Test-Error: 0.336
# Test-Accuracy: 0.843
# Train-Error: 0.244
# Train-Accuracy: 0.902


# --------------------------------------------------
# Batch gradient descent - changing the weights every 100 times and taking the average, this avoids taking the noise and not the signal from an input once it averages out

(x_train, y_train), (x_test, y_test) = mnist.load_data()
images, labels = (x_train[:1000].reshape(1000, 28 * 28) / 255, y_train[:1000])
one_hot_labels = np.zeros((len(labels), 10))

for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

test_images = x_test.reshape(len(x_test), 28 * 28) / 255
test_labels = np.zeros((len(y_test), 10))

for i, l in enumerate(y_test):
    test_labels[i][l] = 1


def relu(x):
    return (x >= 0) * x  # returns x if x > 0


def relu2deriv(output):
    return output >= 0  # returns 1 for input > 0


batch_size = 100
alpha, iterations = (0.001, 300)
pixels_per_image, num_labels, hidden_size = (784, 10, 100)

weights_0_1 = 0.2 * np.random.random((pixels_per_image, hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

for j in range(iterations):
    error, correct_count = (0.0, 0)
    for i in range(int(len(images) / batch_size)):
        batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))

        layer_0 = images[batch_start:batch_end]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1 *= dropout_mask * 2
        layer_2 = np.dot(layer_1, weights_1_2)

        error += np.sum((labels[batch_start:batch_end] - layer_2) ** 2)
        for k in range(batch_size):
            correct_count += int(
                np.argmax(layer_2[k : k + 1])
                == np.argmax(labels[batch_start + k : batch_start + k + 1])
            )

            layer_2_delta = (labels[batch_start:batch_end] - layer_2) / batch_size
            layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)
            layer_1_delta *= dropout_mask

            weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
            weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

    if j % 10 == 0:
        test_error = 0.0
        test_correct_count = 0

        for i in range(len(test_images)):
            layer_0 = test_images[i : i + 1]
            layer_1 = relu(np.dot(layer_0, weights_0_1))
            layer_2 = np.dot(layer_1, weights_1_2)

            test_error += np.sum((test_labels[i : i + 1] - layer_2) ** 2)
            test_correct_count += int(
                np.argmax(layer_2) == np.argmax(test_labels[i : i + 1])
            )

        sys.stdout.write(
            "\r"
            + "I:"
            + str(j)
            + "\nTest-Error: "
            + str(test_error / float(len(test_images)))[:5]
            + "\nTest-Accuracy: "
            + str(test_correct_count / float(len(test_images)))[:5]
            + "\nTrain-Error: "
            + str(error / float(len(images)))[:5]
            + "\nTrain-Accuracy: "
            + str(correct_count / float(len(images)))
        )

# I:290
# Test-Error: 0.421
# Test-Accuracy: 0.808
# Train-Error: 0.397
# Train-Accuracy: 0.84
