# Chapter 4 - gradient descent

# finding out the error

knob_weight = 0.5
input_ = 0.5
goal_pred = 0.8

pred = input_ * knob_weight
error = (pred - goal_pred) ** 2
print(error)
# 0.30


# ------------------------------------------
# hot and cold learning

weight = 0.5
input_ = 0.5
goal_prediction = 0.8

step_amount = 0.001  # how much to move each iteration

# we know that it takes 1100 iterations to predict correctly
for iteration in range(1101):
    prediction = weight * input_
    error = (prediction - goal_prediction) ** 2

    print(f"Error: {error}, Prediction: {prediction}, Weight: {weight}")

    up_prediction = input_ * (weight + step_amount)
    up_error = input_ * (goal_prediction - up_prediction) ** 2

    down_prediction = input_ * (weight - step_amount)
    down_error = input_ * (goal_prediction - down_prediction) ** 2

    if down_error < up_error:
        weight + -step_amount

    elif down_error > up_error:
        weight += step_amount


# Few first lines


# Error: 0.30250000000000005, Prediction: 0.25, Weight: 0.5
# Error: 0.3019502500000001, Prediction: 0.2505, Weight: 0.501
# Error: 0.30140100000000003, Prediction: 0.251, Weight: 0.502
# Error: 0.30085225, Prediction: 0.2515, Weight: 0.503
# Error: 0.30030400000000007, Prediction: 0.252, Weight: 0.504


# Few middle lines

# Error: 0.029929000000004778, Prediction: 0.6269999999999862, Weight: 1.2539999999999725
# Error: 0.029756250000004782, Prediction: 0.6274999999999862, Weight: 1.2549999999999724
# Error: 0.029584000000004787, Prediction: 0.6279999999999861, Weight: 1.2559999999999722
# Error: 0.029412250000004792, Prediction: 0.6284999999999861, Weight: 1.2569999999999721
# Error: 0.029241000000004798, Prediction: 0.628999999999986, Weight: 1.257999999999972


# Few last lines

# Error: 4.000000000130569e-06, Prediction: 0.7979999999999674, Weight: 1.5959999999999348
# Error: 2.2500000000980924e-06, Prediction: 0.7984999999999673, Weight: 1.5969999999999347
# Error: 1.000000000065505e-06, Prediction: 0.7989999999999673, Weight: 1.5979999999999346
# Error: 2.5000000003280753e-07, Prediction: 0.7994999999999672, Weight: 1.5989999999999345
# Error: 1.0799505792475652e-27, Prediction: 0.7999999999999672, Weight: 1.5999999999999344


# ------------------------------------------
# calculating both direction and amount from error

weight = 0.5
goal_prediction = 0.8
input_ = 0.5

for iteration in range(20):
    prediction = input_ * weight
    error = (prediction - goal_prediction) ** 2
    direction_and_amount = (
        prediction - goal_prediction
    ) * input_  # pure error * input_
    weight -= direction_and_amount

    print(f"Error: {error} Prediction: {prediction} Weight: {weight}")


# Error: 0.30250000000000005 Prediction: 0.25 Weight: 0.775
# Error: 0.17015625000000004 Prediction: 0.3875 Weight: 0.9812500000000001
# Error: 0.095712890625 Prediction: 0.49062500000000003 Weight: 1.1359375
# Error: 0.05383850097656251 Prediction: 0.56796875 Weight: 1.251953125
# Error: 0.03028415679931642 Prediction: 0.6259765625 Weight: 1.33896484375
# Error: 0.0170348381996155 Prediction: 0.669482421875 Weight: 1.4042236328125
# Error: 0.00958209648728372 Prediction: 0.70211181640625 Weight: 1.453167724609375
# Error: 0.005389929274097089 Prediction: 0.7265838623046875 Weight: 1.4898757934570312
# Error: 0.0030318352166796153 Prediction: 0.7449378967285156 Weight: 1.5174068450927733
# Error: 0.0017054073093822882 Prediction: 0.7587034225463867 Weight: 1.53805513381958
# Error: 0.0009592916115275371 Prediction: 0.76902756690979 Weight: 1.553541350364685
# Error: 0.0005396015314842384 Prediction: 0.7767706751823426 Weight: 1.5651560127735138
# Error: 0.000303525861459885 Prediction: 0.7825780063867569 Weight: 1.5738670095801353
# Error: 0.00017073329707118678 Prediction: 0.7869335047900676 Weight: 1.5804002571851015
# Error: 9.603747960254256e-05 Prediction: 0.7902001285925507 Weight: 1.5853001928888262
# Error: 5.402108227642978e-05 Prediction: 0.7926500964444131 Weight: 1.5889751446666196
# Error: 3.038685878049206e-05 Prediction: 0.7944875723333098 Weight: 1.5917313584999646
# Error: 1.7092608064027242e-05 Prediction: 0.7958656792499823 Weight: 1.5937985188749735
# Error: 9.614592036015323e-06 Prediction: 0.7968992594374867 Weight: 1.5953488891562302
# Error: 5.408208020258491e-06 Prediction: 0.7976744445781151 Weight: 1.5965116668671726


# Three main attributes of translating the pure error into the absolute amount you want to change the weight are: Stopping, Negative reversal and scaling

# What is stopping?
# Imagine you have a CD player and you turn the volume all the way up but the CD player is off, tha volume change would not matter. If the input is 0, then it will force direction_and_amount to also be 0, when input is 0 you don't learn because there is nothing to learn.

# What is negative reversal?
# It's making sure that weight moves in the correct directio neven if input is negative

# What is scaling?
# It's simply the effect of multiplying the pure error by input. If the input is big, logically your weight update should also be big, it often goes out of control, use alpha to control it.


# ------------------------------------------
# learning is just reducing error

weight = 0
goal_pred = 0.8
input_ = 0.5

for iteration in range(4):
    pred = input_ * weight
    error = (pred - goal_pred) ** 2
    delta = pred - goal_pred
    weight_delta = delta * input_
    weight -= weight_delta

    print(
        f"Error: {error} Prediction: {pred} Weight: {weight} Weight Delta {weight_delta}"
    )


# Error: 0.64 Prediction: 0.0 Weight: 0.4 Weight Delta -0.4
# Error: 0.36 Prediction: 0.2 Weight: 0.70 Weight Delta -0.30
# Error: 0.202 Prediction: 0.35 Weight: 0.925 Weight Delta -0.225
# Error: 0.113 Prediction: 0.4625 Weight: 1.09375 Weight Delta -0.16875