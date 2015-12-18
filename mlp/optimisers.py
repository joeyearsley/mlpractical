# Machine Learning Practical (INFR11119),
# Pawel Swietojanski, University of Edinburgh

import numpy
import time
import logging

from mlp.layers import MLP,Sigmoid
from mlp.dataset import DataProvider
from mlp.schedulers import LearningRateScheduler
from mlp.costs import MSECost

logger = logging.getLogger(__name__)


class Optimiser(object):
    def train_epoch(self, model, train_iter):
        raise NotImplementedError()

    def train(self, model, train_iter, valid_iter=None):
        raise NotImplementedError()

    def validate(self, model, valid_iterator, l1_weight=0, l2_weight=0):
        assert isinstance(model, MLP), (
            "Expected model to be a subclass of 'mlp.layers.MLP'"
            " class but got %s " % type(model)
        )

        assert isinstance(valid_iterator, DataProvider), (
            "Expected iterator to be a subclass of 'mlp.dataset.DataProvider'"
            " class but got %s " % type(valid_iterator)
        )

        acc_list, nll_list = [], []
        for x, t in valid_iterator:
            y = model.fprop(x)
            nll_list.append(model.cost.cost(y, t))
            acc_list.append(numpy.mean(self.classification_accuracy(y, t)))

        acc = numpy.mean(acc_list)
        nll = numpy.mean(nll_list)

        prior_costs = Optimiser.compute_prior_costs(model, l1_weight, l2_weight)

        return nll + sum(prior_costs), acc

    @staticmethod
    def classification_accuracy(y, t):
        """
        Returns classification accuracy given the estimate y and targets t
        :param y: matrix -- estimate produced by the model in fprop
        :param t: matrix -- target  1-of-K coded
        :return: vector of y.shape[0] size with binary values set to 0
                 if example was miscalssified or 1 otherwise
        """
        y_idx = numpy.argmax(y, axis=1)
        t_idx = numpy.argmax(t, axis=1)
        rval = numpy.equal(y_idx, t_idx)
        return rval

    @staticmethod
    def compute_prior_costs(model, l1_weight, l2_weight):
        """
        Computes the cost contributions coming from parameter-dependent only
        regularisation penalties
        """
        assert isinstance(model, MLP), (
            "Expected model to be a subclass of 'mlp.layers.MLP'"
            " class but got %s " % type(model)
        )

        l1_cost, l2_cost = 0, 0
        for i in xrange(0, len(model.layers)):
            params = model.layers[i].get_params()
            for param in params:
                if l2_weight > 0:
                    l2_cost += 0.5 * l2_weight * numpy.sum(param**2)
                if l1_weight > 0:
                    l1_cost += l1_weight * numpy.sum(numpy.abs(param))

        return l1_cost, l2_cost


class SGDOptimiser(Optimiser):
    def __init__(self, lr_scheduler,
                 dp_scheduler=None,
                 l1_weight=0.0,
                 l2_weight=0.0):

        super(SGDOptimiser, self).__init__()

        assert isinstance(lr_scheduler, LearningRateScheduler), (
            "Expected lr_scheduler to be a subclass of 'mlp.schedulers.LearningRateScheduler'"
            " class but got %s " % type(lr_scheduler)
        )

        self.lr_scheduler = lr_scheduler
        self.dp_scheduler = dp_scheduler
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def train_epoch(self, model, train_iterator, learning_rate):

        assert isinstance(model, MLP), (
            "Expected model to be a subclass of 'mlp.layers.MLP'"
            " class but got %s " % type(model)
        )
        assert isinstance(train_iterator, DataProvider), (
            "Expected iterator to be a subclass of 'mlp.dataset.DataProvider'"
            " class but got %s " % type(train_iterator)
        )

        acc_list, nll_list = [], []
        
        #Next epoch, next dropout rate if annealed.
        if self.dp_scheduler is not None:
            self.dp_scheduler.get_next_rate()
        
        for x, t in train_iterator:

            # get the prediction
            if self.dp_scheduler is not None:
                y = model.fprop_dropout(x, self.dp_scheduler)
            else:
                y = model.fprop(x)

            # compute the cost and grad of the cost w.r.t y
            cost = model.cost.cost(y, t)
            cost_grad = model.cost.grad(y, t)

            # do backward pass through the model
            model.bprop(cost_grad, self.dp_scheduler)

            #update the model, here we iterate over layers
            #and then over each parameter in the layer
            effective_learning_rate = learning_rate / x.shape[0]
            
            for i in xrange(0, len(model.layers)):
                params = model.layers[i].get_params()
                grads = model.layers[i].pgrads(inputs=model.activations[i],
                                               deltas=model.deltas[i + 1],
                                               l1_weight=self.l1_weight,
                                               l2_weight=self.l2_weight)
                uparams = []
                for param, grad in zip(params, grads):
                    param = param - effective_learning_rate * grad
                    uparams.append(param)
                model.layers[i].set_params(uparams)

            nll_list.append(cost)
            acc_list.append(numpy.mean(self.classification_accuracy(y, t)))

        #compute the prior penalties contribution (parameter dependent only)
        prior_costs = Optimiser.compute_prior_costs(model, self.l1_weight, self.l2_weight)
        training_cost = numpy.mean(nll_list) + sum(prior_costs)

        return training_cost, numpy.mean(acc_list)

    def train(self, model, train_iterator, valid_iterator=None):

        converged = False
        cost_name = model.cost.get_name()
        tr_stats, valid_stats = [], []

        # do the initial validation
        train_iterator.reset()
        tr_nll, tr_acc = self.validate(model, train_iterator, self.l1_weight, self.l2_weight)
        logger.info('Epoch %i: Training cost (%s) for initial model is %.3f. Accuracy is %.2f%%'
                    % (self.lr_scheduler.epoch, cost_name, tr_nll, tr_acc * 100.))
        tr_stats.append((tr_nll, tr_acc))

        if valid_iterator is not None:
            valid_iterator.reset()
            valid_nll, valid_acc = self.validate(model, valid_iterator, self.l1_weight, self.l2_weight)
            logger.info('Epoch %i: Validation cost (%s) for initial model is %.3f. Accuracy is %.2f%%'
                        % (self.lr_scheduler.epoch, cost_name, valid_nll, valid_acc * 100.))
            valid_stats.append((valid_nll, valid_acc))

        while not converged:
            train_iterator.reset()

            tstart = time.clock()
            tr_nll, tr_acc = self.train_epoch(model=model,
                                              train_iterator=train_iterator,
                                              learning_rate=self.lr_scheduler.get_rate())
            tstop = time.clock()
            tr_stats.append((tr_nll, tr_acc))

            logger.info('Epoch %i: Training cost (%s) is %.3f. Accuracy is %.2f%%'
                        % (self.lr_scheduler.epoch + 1, cost_name, tr_nll, tr_acc * 100.))

            vstart = time.clock()
            if valid_iterator is not None:
                valid_iterator.reset()
                valid_nll, valid_acc = self.validate(model, valid_iterator,
                                                     self.l1_weight, self.l2_weight)
                logger.info('Epoch %i: Validation cost (%s) is %.3f. Accuracy is %.2f%%'
                            % (self.lr_scheduler.epoch + 1, cost_name, valid_nll, valid_acc * 100.))
                self.lr_scheduler.get_next_rate(valid_acc)
                valid_stats.append((valid_nll, valid_acc))
            else:
                self.lr_scheduler.get_next_rate(None)
            vstop = time.clock()

            train_speed = train_iterator.num_examples_presented() / (tstop - tstart)
            valid_speed = valid_iterator.num_examples_presented() / (vstop - vstart)
            tot_time = vstop - tstart
            #pps = presentations per second
            logger.info("Epoch %i: Took %.0f seconds. Training speed %.0f pps. "
                        "Validation speed %.0f pps."
                        % (self.lr_scheduler.epoch, tot_time, train_speed, valid_speed))

            # we stop training when learning rate, as returned by lr scheduler, is 0
            # this is implementation dependent and depending on lr schedule could happen,
            # for example, when max_epochs has been reached or if the progress between
            # two consecutive epochs is too small, etc.
            converged = (self.lr_scheduler.get_rate() == 0)

        return tr_stats, valid_stats
    
    '''
        Pretrain using auto-encoders, go through until final layer training for certain amount of epochs.
        Then fine tune by adding final layer and doing standard backprop.
    '''
    def pretrain(self, model, train_iterator, valid_iterator=None):
        converged = False
        cost_name = model.cost.get_name()
        tr_stats, valid_stats = [], []
        rng = numpy.random.RandomState([2015,10,10])
        #Empty list for inputs, so we keep consistent throughout the stacks
        inputs = []
        
        """
        Do the below for each layer
        """
        
        layers = model.layers
        
        #Train first layer normally, then build on top.
        #Remember to add noise - add option to reset, to with noise, allow gaussian noise to be added.
        #Remember sigmoid cost to implement.
        
        #MSE still, so don't use actual models
        cost = MSECost()
        #define the temporary model
        trainingModel = MLP(cost=cost)
        
        
        '''
            Build up inputs for epochs, only interested in x.
        '''
        
        for x,t in train_iterator:
            inputs.append(x)
        
        '''
            
        '''
        
        for i in range(0,len(layers)-1):
            logger.info("Max epochs %i",self.lr_scheduler.max_epochs)
            self.lr_scheduler.reset()
            logger.info("epochs %i",self.lr_scheduler.epoch)
            converged = False
            
            logger.info("Running")
            '''
                Make new model
                Add layer[i] + output Layer(of size layer[i-1])
                pretrain
                take trained layers params
                model.layers[i].setParams = trainedParams
                Re-iterate until finished
            '''
            # Constant out dimensions is equal to the bias of the current layer we wish to add.
            oudim = layers[i].odim
            #If i-1 is zero we have idim of 784, else we have idim of previous layers odim
            if(i == 0):
                trainingModel.add_layer(Sigmoid(idim=784, odim=oudim, rng=rng))
                trainingModel.add_layer(Sigmoid(idim=oudim, odim=784, rng=rng))
            else:
                #in dimension is previous layers odim, which is equal to the size of the bias array.
                indim = layers[i].idim
                trainingModel.add_layer(Sigmoid(idim=indim, odim=oudim, rng=rng))
                trainingModel.add_layer(Sigmoid(idim=oudim, odim=indim, rng=rng))
            
            #No need to validate in pretrain
            
            while not converged:
                train_iterator.reset()

                tstart = time.clock()
                #Pass inputs instead of train iterator.
                tr_nll, tr_acc = self.pretrain_epoch(model=trainingModel,
                                                  train_iterator=inputs,
                                                  learning_rate=self.lr_scheduler.get_rate(), layer=i)
                tstop = time.clock()
                tr_stats.append((tr_nll, tr_acc))

                logger.info('Epoch %i: Training cost (%s) is %.3f. Accuracy is %.2f%%'
                            % (self.lr_scheduler.epoch + 1, cost_name, tr_nll, tr_acc * 100.))

                # we stop training when learning rate, as returned by lr scheduler, is 0
                # this is implementation dependent and depending on lr schedule could happen,
                # for example, when max_epochs has been reached or if the progress between
                # two consecutive epochs is too small, etc.
                converged = (self.lr_scheduler.get_next_rate(tr_acc) == 0)
            
            
            #Now set parameters of layer we added.
            model.layers[i].set_params(trainingModel.layers[i].get_params())
            
            logger.info('activations %i',len(trainingModel.activations))
            #Remove activation as we aren't interested in the final activation
            trainingModel.activations = trainingModel.activations[:-1]
            logger.info('activations2 %i',len(trainingModel.activations))
            
            #Now remove final layer as we aren't interested in it.
            trainingModel.set_layers(trainingModel.layers[:-1])
                
        self.lr_scheduler.reset()
                    
        #Return at end?
        return tr_stats, valid_stats
        

    def pretrain_epoch(self, model, train_iterator, learning_rate, layer):

        assert isinstance(model, MLP), (
            "Expected model to be a subclass of 'mlp.layers.MLP'"
            " class but got %s " % type(model)
        )
        '''
        assert isinstance(train_iterator, DataProvider), (
            "Expected iterator to be a subclass of 'mlp.dataset.DataProvider'"
            " class but got %s " % type(train_iterator)
        )
        '''
        acc_list, nll_list = [], []
        
        #Next epoch, next dropout rate if annealed.
        if self.dp_scheduler is not None:
            self.dp_scheduler.get_next_rate()
        
        #Do this for max_epochs?
        #Move to the function to pretrain to ensure same images, then just pass the correct list each epoch.
        #Replacing the x,t
        for x in train_iterator:
            
            #Add Noise Here to the image given, if it is the first layer, else just put it into xhat.
            
            
            
            
            # get the prediction
            if self.dp_scheduler is not None:
                y = model.fprop_dropout(x, self.dp_scheduler)
            else:
                y = model.fprop(x)
            
            #If layer is not 0, as we want to do this after the intial input
            if layer != 0:
                x = model.activations[layer]
                
            #logger.info(model.activations[layer].shape)
            #logger.info(x.shape)
            
            # compute the cost and grad of the cost w.r.t y
            cost = model.cost.cost(y, x)
            cost_grad = model.cost.grad(y, x)

            # do backward pass through the 2 layers we are interested in.
            model.bprop(cost_grad, self.dp_scheduler, pretrain=True)
            
            #update the model, here we iterate over layers
            #and then over each parameter in the layer
            effective_learning_rate = learning_rate / x.shape[0]
            
            #Loop through current layer and end
            for i in xrange(len(model.layers)-2, len(model.layers)):
                params = model.layers[i].get_params()
                grads = model.layers[i].pgrads(inputs=model.activations[i],
                                               deltas=model.deltas[i + 1],
                                               l1_weight=self.l1_weight,
                                               l2_weight=self.l2_weight)
                uparams = []
                for param, grad in zip(params, grads):
                    param = param - effective_learning_rate * grad
                    uparams.append(param)
                model.layers[i].set_params(uparams)

            nll_list.append(cost)
            acc_list.append(numpy.mean(self.classification_accuracy(y, x)))

        #compute the prior penalties contribution (parameter dependent only)
        prior_costs = Optimiser.compute_prior_costs(model, self.l1_weight, self.l2_weight)
        training_cost = numpy.mean(nll_list) + sum(prior_costs)

        return training_cost, numpy.mean(acc_list)
