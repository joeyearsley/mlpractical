# %load Experiments/scheduler.py
#Baseline experiment

from mlp.layers import MLP, Linear, Sigmoid, Softmax #import required layer types
from mlp.optimisers import SGDOptimiser #import the optimiser

from mlp.costs import CECost #import the cost we want to use for optimisation
from mlp.schedulers import LearningRateExponential, LearningRateFixed, LearningRateList, LearningRateNewBob, DropoutAnnealed

import numpy
import logging
import shelve
from mlp.dataset import MNISTDataProvider

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info('Initialising data providers...')

train_dp = MNISTDataProvider(dset='train', batch_size=100, max_num_batches=1000, randomize=True)
valid_dp = MNISTDataProvider(dset='valid', batch_size=10000, max_num_batches=-10, randomize=False)
test_dp = MNISTDataProvider(dset='eval', batch_size=10000, max_num_batches=-10, randomize=False)

rng = numpy.random.RandomState([2015,10,10])

#some hyper-parameters
nhid = 800
max_epochs = 50
cost = CECost()
learning_rate = 0.5;
learningList = []
decrement = (learning_rate/max_epochs)

#Regulariser weights
l1_weight = 0.000
l2_weight = 0.000

#Build list once so we don't have to rebuild every time.
for i in xrange(0,max_epochs):
    #In this order so start learning rate is added
    learningList.append(learning_rate)
    learning_rate -= decrement



#Open file to save to
shelve_r = shelve.open("regExperiments")

stats = []
rate = 1

#For each number of layers, new model add layers.
for layer in xrange(0,4):
    
    #Reset
    dp_scheduler = DropoutAnnealed(0.5, 0.5, 0.05)
    
    #Set here in case we alter it in a layer experiment
    learning_rate = 0.5


    train_dp.reset()
    valid_dp.reset()
    test_dp.reset()

    logger.info("Starting ")

    #define the model
    model = MLP(cost=cost)

    if layer == 0:
        odim = 800
        model.add_layer(Sigmoid(idim=784, odim=odim, irange=0.2, rng=rng))
    elif layer == 1:
        odim = 600
        model.add_layer(Sigmoid(idim=784, odim=600, irange=0.2, rng=rng))
        model.add_layer(Sigmoid(idim=600, odim=600, irange=0.2, rng=rng))
    elif layer == 2:
        odim = 400
        model.add_layer(Sigmoid(idim=784, odim=odim, irange=0.2, rng=rng))
        model.add_layer(Sigmoid(idim=odim, odim=odim, irange=0.2, rng=rng))
        model.add_layer(Sigmoid(idim=odim, odim=odim, irange=0.2, rng=rng))
    elif layer == 3:
        odim = 300
        model.add_layer(Sigmoid(idim=784, odim=odim, irange=0.2, rng=rng))
        model.add_layer(Sigmoid(idim=odim, odim=odim, irange=0.2, rng=rng))
        model.add_layer(Sigmoid(idim=odim, odim=odim, irange=0.2, rng=rng))
        model.add_layer(Sigmoid(idim=odim, odim=odim, irange=0.2, rng=rng))
        
    #Add output layer
    model.add_layer(Softmax(idim=odim, odim=10, rng=rng))

    #Set rate scheduler here
    if rate == 1:
        lr_scheduler = LearningRateExponential(start_rate=learning_rate, max_epochs=max_epochs, training_size=100)
    elif rate == 2:
        lr_scheduler = LearningRateFixed(learning_rate=learning_rate, max_epochs=max_epochs)
    elif rate == 3:
        # define the optimiser, here stochasitc gradient descent
        # with fixed learning rate and max_epochs
        lr_scheduler = LearningRateNewBob(start_rate=learning_rate, max_epochs=max_epochs,\
                                          min_derror_stop=.05, scale_by=0.05, zero_rate=learning_rate, patience = 10)

    optimiser = SGDOptimiser(lr_scheduler=lr_scheduler, 
                             dp_scheduler=dp_scheduler,
                             l1_weight=l1_weight, 
                             l2_weight=l2_weight)

    logger.info('Training started...')
    tr_stats, valid_stats = optimiser.train(model, train_dp, valid_dp)

    logger.info('Testing the model on test set:')
    tst_cost, tst_accuracy = optimiser.validate(model, test_dp)
    logger.info('MNIST test set accuracy is %.2f %%, cost (%s) is %.3f'%(tst_accuracy*100., cost.get_name(), tst_cost))

    #Append stats for all test
    stats.append((tr_stats, valid_stats, (tst_cost, tst_accuracy)))

    #Should save rate to specific dictionairy in pickle, different key so same shelving doesn't matter
    shelve_r['dropA10E'+str(layer)] = (tr_stats, valid_stats, (tst_cost, tst_accuracy))

logger.info('Saving Data')
shelve_r.close()   