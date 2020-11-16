from uwnet import *

def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
#
# The structure of my covnet was defined as the following:
#     covnet_layer1: 32, 32, 3, 8, 3, 1
#     covnet_layer2: 16, 16, 8, 16, 3, 1
#     covnet_layer3: 8, 8, 16, 32, 3, 1
#     covnet_layer4: 4, 4, 32, 64, 3, 1
#     linear_connected_layer: 256, 10
#      
#    Which has the total number of operation calculated by  filter * (kernel/stride)^2 * c * w * h  for each layer
#       = (32 * 32 * 3 * 8 * (3)^2) + (16 * 16 * 8 * 16 * (3)^2) + (8 * 8 * 16 * 32 * (3)^2) + (4 * 4 * 32 * 64 * (3)^2)
#          + (256 * 10)
#       = 1,108,480 operations
#   The final accuratecy are training accuracy: 62.36%
#                                test accuracy: 59.87%
# 
# While the structure of my fully connected model was defined as the following:
#    layer1: 3072, 350
#    layer2: 350, 350
#    layer3: 350, 10
# 
#    Which has the total number of operation calculated by input * output for each layer
#       =  3072*300 + 300 * 300 + 300*10 = 1,201,200 operations
#   The final accuratecy training accuracy: 48.38%
#                            test accuracy: 46.10%
# 
# 
#   From the above experiment, given similar number of operations, covnet has outperformed
# the fully connected network model. I believe that the reason behind is that when we apply
# kernels onto the the convnet, the relationship of spatiality of those pixels that 
# are relatively closer to each other becomes more important than the ones that are not. 
# As an oppose, the fully connected network does not take this relationship of 
# spitiality into account, in fact every relationship has the same importance for evaluation. 