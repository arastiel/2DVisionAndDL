# Exercise02

# Resources:
# https://cs231n.github.io/convolutional-networks/#pool

# It is common to periodically insert a Pooling layer in-between successive Conv layers in a ConvNet architecture.
# Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters
# and computation in the network, and hence to also control overfitting. The Pooling Layer operates independently on
# every depth slice of the input and resizes it spatially, using the MAX operation. The most common form is a pooling
# layer with filters of size 2x2 applied with a stride of 2 downsamples every depth slice in the input by 2 along both
# width and height, discarding 75% of the activations. Every MAX operation would in this case be taking a max over 4
# numbers (little 2x2 region in some depth slice). The depth dimension remains unchanged.
#
# More generally, the pooling layer:
#
# - Accepts a volume of size W1×H1×D1
# - Requires two hyper parameters:
#
#   => their spatial extent F
#   => the stride S
#
# - Produces a volume of size W2×H2×D2 where:
#
#   => W2=(W1−F)/S+1
#   => H2=(H1−F)/S+1
#   => D2=D1
#
# - Introduces zero parameters since it computes a fixed function of the input
# - For Pooling layers, it is not common to pad the input using zero-padding.

# Strategy for Backpropagation with max pooling:

# The backward pass for a max(x, y) operation has a simple interpretation as only routing the gradient to the input that
# had the highest value in the forward pass. Hence, during the forward pass of a pooling layer it is common to keep
# track of the index of the max activation (sometimes also called the switches) so that gradient routing is efficient
# during backpropagation.
