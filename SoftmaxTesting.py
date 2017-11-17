import numpy
from layers import Softmax

a = numpy.asarray([-20.1, 52.4, 0, 0.05, 0.05, 49])
b = numpy.asarray([0, 0, 0, 0, 0, 0, 0, 1])
rng = numpy.random.RandomState([2015,10,10])
rng_state = rng.get_state()

rng.set_state(rng_state)
softmax = Softmax(inputDimensions=a.shape[0], outputDimensions=b.shape[0], rng=rng)

fp = softmax.fprop(a)
deltas, ograds = softmax.bprop_cost(h=None, igrads=fp-b, cost=None)

print(fp.sum())
print(deltas.sum())
print(ograds.sum())
print(fp)
print(deltas)
print(ograds)