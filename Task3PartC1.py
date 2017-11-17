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

weightMatrixSigmoid = sigmoidLayer.get_params()[0]
numpyWeights = numpy.matrix(weightMatrixSigmoid)

print(weightMatrixSigmoid.shape)
print(numpyWeights.shape)



fig = plt.figure()
fif, axarr = plt.subplots(10, 10)

arrayOfMatrices = []
for i in range(0, 100):
    singleHiddenWeight = numpyWeights[:, i]
    #print(singleHiddenWeight.shape)
    reshapedHiddenWeights = numpy.reshape(singleHiddenWeight, (28, 28))
    arrayOfMatrices.insert(i, reshapedHiddenWeights)
    m = i%10;
    l = i // 10
    axarr[l,m].imshow(numpy.reshape(singleHiddenWeight, (28, 28)), cmap=cm.Greys_r)
    print(reshapedHiddenWeights)


plt.show()


