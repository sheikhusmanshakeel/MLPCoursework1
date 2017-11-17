import numpy
import logging

logging.basicConfig()
logger = logging.getLogger('My Logger')
logger.setLevel(logging.INFO)

logger.info("write something here")

from layers import MLP, Linear, Sigmoid, Softmax  # import required layer types
from optimisers import SGDOptimiser  # import the optimiser
from dataset import MNISTDataProvider  # import data provider
from costs import MSECost, CECost  # import the cost we want to use for optimisation
from schedulers import LearningRateFixed
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2**numpy.ceil(numpy.log(numpy.abs(matrix).max())/numpy.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x,y),w in numpy.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = numpy.sqrt(numpy.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()

rng = numpy.random.RandomState([2015, 10, 10])

# define the model structure, here just one linear layer
# and mean square error cost
# cost = MSECost()
cost = CECost()
model = MLP(cost=cost)

linearLayer = Linear(idim=784, odim=784, rng=rng)
sigmoidLayer = Sigmoid(inputDimensions=784, outputDimensions=100, rng=rng)
softmaxLayer = Softmax(inputDimensions=100, outputDimensions=10, rng=rng)

model.add_layer(linearLayer)
model.add_layer(sigmoidLayer)
model.add_layer(softmaxLayer)

# one can stack more layers here
maxBatches = 300
batchSize = 10
numberOfEpochs = 10
# define the optimiser, here stochasitc gradient descent
# with fixed learning rate and max_epochs as stopping criterion
lr_scheduler = LearningRateFixed(learning_rate=0.01, max_epochs=numberOfEpochs)
optimiser = SGDOptimiser(lr_scheduler=lr_scheduler)

logger.info('Initialising data providers...')
train_dp = MNISTDataProvider(dset='train', batch_size=batchSize, max_num_batches=maxBatches, randomize=True)
valid_dp = MNISTDataProvider(dset='valid', batch_size=batchSize, max_num_batches=maxBatches, randomize=False)

logger.info('Training started...')
trainingStats, validStats = optimiser.train(model, train_dp, valid_dp)

logger.info('Testing the model on test set:')
test_dp = MNISTDataProvider(dset='eval', batch_size=100, max_num_batches=-10, randomize=False)
cost, accuracy = optimiser.validate(model, test_dp)
logger.info('MNIST test set accuracy is %.2f %% (cost is %.3f)' % (accuracy * 100., cost))

weightMatrixSoftmax = numpy.matrix(softmaxLayer.get_params()[0])
print(weightMatrixSoftmax.shape)
weightMatrixSoftmaxRow0 = weightMatrixSoftmax[:,0 ]
weightMatrixSoftmaxRow1 = weightMatrixSoftmax[:,1 ]

print(weightMatrixSoftmaxRow0.shape)
print(weightMatrixSoftmaxRow1.shape)


#print(weightMatrixSigmoid.shape)
#print(numpyWeights.shape)

hinton(weightMatrixSoftmaxRow0.T)
plt.title('Hinton diagram for weights 0')
plt.show()


hinton(weightMatrixSoftmaxRow1.T)
plt.title('Hinton diagram for weights 1')
plt.show()


