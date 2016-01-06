# Machine Learning Practical (INFR11119),
# Pawel Swietojanski, University of Edinburgh

import logging
import numpy

logger = logging.getLogger(__name__)

class LearningRateScheduler(object):
    """
    Define an interface for determining learning rates
    """
    def __init__(self, max_epochs=100):
        self.epoch = 0
        self.max_epochs = max_epochs

    def get_rate(self):
        raise NotImplementedError()

    def get_next_rate(self, current_accuracy=None):
        self.epoch += 1
        
    #Reset the epochs for pretrain, ensures just one method can be called
    def reset(self):
        self.epoch = 0
       
        
class LearningRateList(LearningRateScheduler):
    def __init__(self, learning_rates_list, max_epochs):

        super(LearningRateList, self).__init__(max_epochs)

        assert isinstance(learning_rates_list, list), (
            "The learning_rates_list argument expected"
            " to be of type list, got %s" % type(learning_rates_list)
        )
        self.lr_list = learning_rates_list
        
    def get_rate(self):
        if self.epoch < len(self.lr_list):
            return self.lr_list[self.epoch]
        return 0.0
    
    def get_next_rate(self, current_accuracy=None):
        super(LearningRateList, self).get_next_rate(current_accuracy=None)
        return self.get_rate()
         
class LearningRateFixed(LearningRateList):

    def __init__(self, learning_rate, max_epochs):
        assert learning_rate > 0, (
            "learning rate expected to be > 0, got %f" % learning_rate
        )
        super(LearningRateFixed, self).__init__([learning_rate], max_epochs)

    def get_rate(self):
        if self.epoch < self.max_epochs:
            return self.lr_list[0]
        return 0.0

    def get_next_rate(self, current_accuracy=None):
        super(LearningRateFixed, self).get_next_rate(current_accuracy=None)
        return self.get_rate()

class LearningRateExponential(LearningRateScheduler):
    '''
        Exponentially decreasing learning rate.
        zero_rate - rate to multiply the epoch/training size by, keyword argument
    '''
    def __init__(self, start_rate, max_epochs, training_size, zero_rate=0.5):
        
        #Set the training size
        self.training_size = training_size
        
        #Do checks as both need to be greater than zero
        assert start_rate > 0, (
            "starting rate expected to be > 0, got %f" % start_rate
        )
        assert zero_rate > 0, (
            "zero rate expected to be > 0, got %f" % zero_rate
        )
        
        #Init the super class with the max epochs
        super(LearningRateExponential, self).__init__(max_epochs)
        
        #Set the class properties
        self.start_rate = start_rate
        self.zero_rate = zero_rate
        self.rate = start_rate
        self.epoch = 1
    
    #Reset the epochs and start_rate
    def reset(self):
        self.rate = self.start_rate
        self.epoch = 0
        
    #Return the current rate
    def get_rate(self):
        return self.rate  
    
    def get_next_rate(self,current_accuracy=None):  
        # If epochs have over ran return zero
        if ( (self.max_epochs > 10000) or (self.epoch >= self.max_epochs) ):
            self.rate = 0.0
        else:
            #Use float or it won't return properly.
            self.rate = self.zero_rate * numpy.exp(-float(self.epoch)/float(self.training_size))
            #Increase the epochs for the next round
            self.epoch += 1
        #Log the rate for checking in logger - Not entirely needed
        logger.info(self.rate)
        return self.rate


    
    
class LearningRateNewBob(LearningRateScheduler):
    """
    newbob learning rate schedule.
    
    Fixed learning rate until validation set stops improving then exponential
    decay.
    """
    
    def __init__(self, start_rate, scale_by=.5, max_epochs=99,
                 min_derror_ramp_start=.5, min_derror_stop=.5, init_error=100.0,
                 patience=0, zero_rate=None, ramping=False):
        """
        :type start_rate: float
        :param start_rate: 
        
        :type scale_by: float
        :param scale_by: 
        
        :type max_epochs: int
        :param max_epochs: 
        
        :type min_error_start: float
        :param min_error_start: 
        
        :type min_error_stop: float
        :param min_error_stop: 
        
        :type init_error: float
        :param init_error: 
        """
        self.start_rate = start_rate
        self.init_error = init_error
        self.init_patience = patience
        
        self.rate = start_rate
        self.scale_by = scale_by
        self.max_epochs = max_epochs
        self.min_derror_ramp_start = min_derror_ramp_start
        self.min_derror_stop = min_derror_stop
        self.lowest_error = init_error
        
        self.epoch = 1
        self.ramping = ramping
        self.patience = patience
        self.zero_rate = zero_rate
        
    def reset(self):
        self.rate = self.start_rate
        self.lowest_error = self.init_error
        self.epoch = 1
        self.ramping = False
        self.patience = self.init_patience
    
    def get_rate(self):
        if (self.epoch==1 and self.zero_rate!=None):
            return self.zero_rate
        return self.rate  
    
    def get_next_rate(self, current_accuracy):
        """
        :type current_accuracy: float
        :param current_accuracy: current proportion correctly classified
        
        """
        
        current_error = 1. - current_accuracy
        diff_error = 0.0
        
        if ( (self.max_epochs > 10000) or (self.epoch >= self.max_epochs) ):
            #logging.debug('Setting rate to 0.0. max_epochs or epoch>=max_epochs')
            self.rate = 0.0
        else:
            diff_error = self.lowest_error - current_error
            
            if (current_error < self.lowest_error):
                self.lowest_error = current_error
    
            if (self.ramping):
                if (diff_error < self.min_derror_stop):
                    if (self.patience > 0):
                        #logging.debug('Patience decreased to %f' % self.patience)
                        self.patience -= 1
                        self.rate *= self.scale_by
                    else:
                        #logging.debug('diff_error (%f) < min_derror_stop (%f)' % (diff_error, self.min_derror_stop))
                        self.rate = 0.0
                else:
                    self.rate *= self.scale_by
            else:
                if (diff_error < self.min_derror_ramp_start):
                    #logging.debug('Start ramping.')
                    self.ramping = True
                    self.rate *= self.scale_by
            
            self.epoch += 1
    
        return self.rate


class DropoutFixed(LearningRateList):

    def __init__(self, p_inp_keep, p_hid_keep):
        assert 0 < p_inp_keep <= 1 and 0 < p_hid_keep <= 1, (
            "Dropout 'keep' probabilites are suppose to be in (0, 1] range"
        )
        super(DropoutFixed, self).__init__([(p_inp_keep, p_hid_keep)], max_epochs=999)

    def get_rate(self):
        return self.lr_list[0]

    def get_next_rate(self, current_accuracy=None):
        return self.get_rate()
    

class DropoutAnnealed(LearningRateList):
    '''

    Increase till 1, extends learning rate list to keep dropout values in order.
    Increases till 1, so that when other pieces of code do:
        p_inp_keep * layer will always return the layer.
    
    '''
    def __init__(self, p_inp_keep, p_hid_keep, constant_to_reduce):
        '''
        
            :type p_inp_keep: float
            :param p_inp_keep: Initial input layers' probability of dropout
            
            :type p_hid_keep: float
            :param p_hid_keep: Initial hidden layers' probability of dropout
            
            :type constant_to_reduce: float
            :param constant_to_reduce: Constant by which to increase at each epoch, until we reach 1
        
        '''
        
        #Assertion to ensure probabilities are good to use
        assert 0 < p_inp_keep <= 1 and 0 < p_hid_keep <= 1, (
            "Dropout 'keep' probabilites are suppose to be in (0, 1] range"
        )
        
        self.lr_temp = []
        
        '''
            To build up the rates, if the rates are set differently.
        '''
        if p_inp_keep > p_hid_keep:
            #Ensures we stop at one.
            while(p_hid_keep < 1):
                #Do in this order to stop probabilities larger than 1 being appended.
                self.lr_temp.append((p_inp_keep, p_hid_keep))
                '''
                    Want to check that the input prob is still under one, if we add to it
                    if it's not then we set to the upper bound of 1.
                '''
                if (p_inp_keep + constant_to_reduce) < 1:
                    p_inp_keep = p_inp_keep + constant_to_reduce
                else:
                    p_inp_keep = 1
                p_hid_keep = p_hid_keep + constant_to_reduce
        elif p_inp_keep < p_hid_keep:
            while(p_inp_keep < 1):
                #Do in this order to stop probabilities larger than 1 being appended.
                self.lr_temp.append((p_inp_keep, p_hid_keep))
                '''
                    Want to check that the hidden prob is still under one, if we add to it
                    if it's not then we set to the upper bound of 1.
                '''
                if (p_hid_keep + constant_to_reduce) < 1:
                    p_hid_keep = p_hid_keep + constant_to_reduce
                else:
                    p_hid_keep = 1
                p_inp_keep = p_inp_keep + constant_to_reduce
        else:
            #Build normally, as they are set together
            while(p_inp_keep < 1 and p_hid_keep < 1):
                #Do in this order to stop probabilities larger than 1 being appended.
                self.lr_temp.append((p_inp_keep, p_hid_keep))
                #Drop out annealed is supposed to increase by a constant amount.
                p_inp_keep = p_inp_keep + constant_to_reduce
                p_hid_keep = p_hid_keep + constant_to_reduce
        
        # Add the upperbounds anyway, as this is what the last one *should* be.
        self.lr_temp.append((1, 1))
        
        super(DropoutAnnealed, self).__init__(self.lr_temp, max_epochs=999)
        
      
    def get_rate(self):
        #Return the current dropout rate
        return self.lr_list[self.epoch]

    '''
        The reason we altered the optimisers
    '''
    def get_next_rate(self, current_accuracy=None):
        # Not ran out of rates yet so return
        if(self.epoch == len(self.lr_list)-1):
            return self.lr_list[self.epoch]
        # Increase epochs so the next rate is used.
        self.epoch += 1
        #Returns the last rate
        return self.lr_list[self.epoch]
