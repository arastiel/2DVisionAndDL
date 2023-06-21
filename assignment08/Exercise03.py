import numpy as np

input = [17, 4, 42, 15, 8, 32, 27]

# Calculate e^int for each int in input
exps = [np.exp(i) for i in input]

sum_of_exps = sum(exps)
# Calculate probability for each int in input using the softmax function
softmax = [j/sum_of_exps for j in exps]

# Print all probabilities in list
print("Probabilites: ", softmax)

# Sum of probabilities is supposed to be 1 (rounding errors may occur)
print("Sum of all probabilities: ", sum(softmax))
# 2p