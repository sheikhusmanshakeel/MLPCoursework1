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

learningRate = [0.5, 0.2, 0.1, 0.05, 0.01, 0.005]
# learningRate = [0.5, 0.2]
aggregatedArrayTraining = []
aggregatedArrayValidation = []
loopCounter = 0

for rate in learningRate:
    # print(rate)
    cost = CECost()
    model = MLP(cost=cost)

    linearLayer = Linear(idim=784, odim=784, rng=rng)
    sigmoidLayer = Sigmoid(inputDimensions=784, outputDimensions=100, rng=rng)
    softmaxLayer = Softmax(inputDimensions=100, outputDimensions=10, rng=rng)

    model.add_layer(linearLayer)
    model.add_layer(sigmoidLayer)
    model.add_layer(softmaxLayer)
    lr_scheduler = LearningRateFixed(learning_rate=rate, max_epochs=20)
    optimiser = SGDOptimiser(lr_scheduler=lr_scheduler)

    logger.info('Initialising data providers...')
    train_dp = MNISTDataProvider(dset='train', batch_size=100, max_num_batches=-10, randomize=True)
    valid_dp = MNISTDataProvider(dset='valid', batch_size=100, max_num_batches=-10, randomize=False)

    logger.info('Training started...')
    trainingStats, validStats = optimiser.train(model, train_dp, valid_dp)

    logger.info('Testing the model on test set:')
    test_dp = MNISTDataProvider(dset='eval', batch_size=100, max_num_batches=-10, randomize=False)
    cost, accuracy = optimiser.validate(model, test_dp)
    logger.info('MNIST test set accuracy is %.2f %% (cost is %.3f)' % (accuracy * 100., cost))

    numpyArrayTraining = numpy.array(trainingStats)
    numpyArrayValidation = numpy.array(validStats)
    intermediateArrayTraining = 1. - numpyArrayTraining[:, 1]
    intermediateArrayValidation = 1.0 - numpyArrayValidation[:,1]
    aggregatedArrayTraining.insert(loopCounter, intermediateArrayTraining)
    aggregatedArrayValidation.insert(loopCounter,intermediateArrayValidation)
    loopCounter += 1

for counter in range(0, len(learningRate)):
    plt.plot(aggregatedArrayTraining[counter], label='LR = {0}'.format(learningRate[counter]))

plt.show()

for counter in range(0, len(learningRate)):
    plt.plot(aggregatedArrayValidation[counter], label='LR = {0}'.format(learningRate[counter]))

plt.show()

