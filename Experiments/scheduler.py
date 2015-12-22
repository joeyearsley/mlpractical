#Baseline experiment

from mlp.layers import MLP, Linear, Sigmoid, Softmax #import required layer types
from mlp.optimisers import SGDOptimiser #import the optimiser

from mlp.costs import CECost #import the cost we want to use for optimisation
from mlp.schedulers import LearningRateExponential, LearningRateFixed, LearningRateList, LearningRateNewBob

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
max_epochs = 5
cost = CECost()

#Open file to save to
shelve_p = shelve.open("learningRateExperiments")

stats = []
#Go through for each learning rate
for rate in xrange(1, 5):

    train_dp.reset()
    valid_dp.reset()
    test_dp.reset()
    
    #define the model
    model = MLP(cost=cost)
    model.add_layer(Sigmoid(idim=784, odim=nhid, irange=0.2, rng=rng))
    model.add_layer(Softmax(idim=nhid, odim=10, rng=rng))
    
    #Set rate scheduler here
    if rate == 1:
        lr_scheduler = LearningRateExponential(start_rate=0.5, max_epochs=max_epochs, training_size=100)
    elif rate == 2:
        lr_scheduler = LearningRateFixed(learning_rate=0.5, max_epochs=max_epochs)
    elif rate == 3:
        # define the optimiser, here stochasitc gradient descent
        # with fixed learning rate and max_epochs
        lr_scheduler = LearningRateNewBob(start_rate=0.5, max_epochs=max_epochs,\
                                      min_derror_stop=.05, scale_by=0.05, zero_rate=0.5, patience = 10)
    elif rate == 4:
        # define the optimiser, here stochasitc gradient descent
        # with fixed learning rate and max_epochs
        lr_scheduler = LearningRateList([0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0.005],max_epochs)
    
    optimiser = SGDOptimiser(lr_scheduler=lr_scheduler)

    logger.info('Training started...')
    tr_stats, valid_stats = optimiser.train(model, train_dp, valid_dp)

    logger.info('Testing the model on test set:')
    tst_cost, tst_accuracy = optimiser.validate(model, test_dp)
    logger.info('MNIST test set accuracy is %.2f %%, cost (%s) is %.3f'%(tst_accuracy*100., cost.get_name(), tst_cost))
    
    #Append stats for all test
    stats.append((tr_stats, valid_stats, (tst_cost, tst_accuracy)))
    
    if rate == 1:
        shelve_p['exponential'] = (tr_stats, valid_stats, (tst_cost, tst_accuracy))
    elif rate == 2:
        shelve_p['fixed'] = (tr_stats, valid_stats, (tst_cost, tst_accuracy))
    elif rate == 3:
        shelve_p['newbob'] = (tr_stats, valid_stats, (tst_cost, tst_accuracy))
    elif rate == 4:
        shelve_p['list'] = (tr_stats, valid_stats, (tst_cost, tst_accuracy))
        
        
shelve_p.close()    