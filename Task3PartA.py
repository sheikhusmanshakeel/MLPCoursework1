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
numberOfEpochs = 5
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

weightMatrixSigmoid = sigmoidLayer.get_params()[0]
numpyWeights = numpy.matrix(weightMatrixSigmoid)

numpyArrayTraining = numpy.array(trainingStats)
errorStats = numpyArrayTraining[:, 0]
finalError = errorStats[len(errorStats) - 1]
print('Final Error: ', finalError)
accuracyStats = numpyArrayTraining[:, 1]
finalAccuracy = accuracyStats[len(accuracyStats) - 1] * 100
print('Final Accuracy: ', finalAccuracy)

# print(weightMatrixSigmoid.shape)
# print(numpyWeights.shape)




# epochs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# accuracy = [1,2,3,4,5,55,6,7,456,3,23,7,8,54,3,2,243,23]


# weightMatrixOfSoftmax = softmaxLayer.get_params()[0]

# print(weightMatrixOfSoftmax)


# plt.plot(epochs,accuracy)
# plt.show()
