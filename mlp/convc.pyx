
# Machine Learning Practical (INFR11119),
# Pawel Swietojanski, University of Edinburgh
import numpy
cimport numpy
import logging
cimport numpy
from mlp.layers import Layer
from types import *




logger = logging.getLogger(__name__)

"""
You have been given some very initial skeleton below. Feel free to build on top of it and/or
modify it according to your needs. Just notice, you can factor out the convolution code out of
the layer code, and just pass (possibly) different conv implementations for each of the stages
in the model where you are expected to apply the convolutional operator. This will allow you to
keep the layer implementation independent of conv operator implementation, and you can easily
swap it layer, for example, for more efficient implementation if you came up with one, etc.
"""

def my1_conv2d(image, kernels, strides=(1, 1)):
    """
    Implements a 2d valid convolution of kernels with the image
    Note: filter means the same as kernel and convolution (correlation) of those with the input space
    produces feature maps (sometimes refered to also as receptive fields). Also note, that
    feature maps are synonyms here to channels, and as such num_inp_channels == num_inp_feat_maps
    :param image: 4D tensor of sizes (batch_size, num_input_channels, img_shape_x, img_shape_y)
    :param filters: 4D tensor of filters of size (num_inp_feat_maps, num_out_feat_maps, kernel_shape_x, kernel_shape_y)
    :param strides: a tuple (stride_x, stride_y), specifying the shift of the kernels in x and y dimensions
    :return: 4D tensor of size (batch_size, num_out_feature_maps, feature_map_shape_x, feature_map_shape_y)
    """
   
    #Get batch_size out
    cdef int batch_size = image.shape[0]
    
    #Get the kernels size
    cdef int out = kernels.shape[1]
    
    '''
    
    Image dims - Used equation from stanford lectures: http://cs231n.github.io/convolutional-networks/
    if((InputSize-FilterSize+2Padding)/Stride+1) is valid int then carry on else throw and error, first calculate
    '''
    
    #Padding not implemented
    cdef int padding = 0
    
    #Get kernel sizes
    cdef int kxdim = kernels.shape[2]
    cdef int kydim = kernels.shape[3]
    
    #Can calculate here as this is all we are going to go to anyway.
    cdef int xdims = ((image.shape[2] - kxdim + (2*padding))/strides[0])+1
    cdef int ydims = ((image.shape[3] - kydim + (2*padding))/strides[1])+1
    
    #Do assertions to ensure passed in the correct type.
    assert type(xdims) is IntType,"Can't make feature map with x-stride: %r" % strides[0]
    assert type(ydims) is IntType,"Can't make feature map with y-stride: %r" % strides[1]
    
    #Get feature map size out
    cdef int num_out_feat_maps = kernels.shape[1]
    #Get input feature size
    cdef int num_inp_feat_maps = kernels.shape[0]
     
    #Create empty 4D tensor
    cdef numpy.ndarray output =  numpy.zeros((batch_size,out,xdims,ydims))
    
    #Declare for speed up
    cdef int img, fm, x, y
    
    #For each image in batch
    for img in xrange(batch_size):
        #For each feature map (output map)
        for fm in xrange(num_out_feat_maps):
            #For each x-dim in output
            for x in xrange(xdims):
                #For each y-dim in output
                for y in xrange(ydims):
                    #Get image slice from entire image, corresponds to kernel size, accross all channels.
                    imgSlice = image[img, :, x:x+kxdim, y:y+kydim]
                    #Get kernels accross all channels.
                    kernel = kernels[:, fm, :, :]
                    '''
                    Do the dot product to get the position.
                    Bias is added in the class.
                    '''
                    output[img, fm, x, y] = numpy.dot(imgSlice.flatten(),kernel.flatten())
                
    return output

class ConvLinear(Layer):
    def __init__(self,
                 num_inp_feat_maps,
                 num_out_feat_maps,
                 image_shape=(28, 28),
                 kernel_shape=(5, 5),
                 stride=(1, 1),
                 irange=0.2,
                 rng=None,
                 conv_fwd=my1_conv2d,
                 conv_bck=my1_conv2d,
                 conv_grad=my1_conv2d):
        """

        :param num_inp_feat_maps: int, a number of input feature maps (channels)
        :param num_out_feat_maps: int, a number of output feature maps (channels)
        :param image_shape: tuple, a shape of the image
        :param kernel_shape: tuple, a shape of the kernel
        :param stride: tuple, shift of kernels in both dimensions
        :param irange: float, initial range of the parameters
        :param rng: RandomState object, random number generator
        :param conv_fwd: handle to a convolution function used in fwd-prop
        :param conv_bck: handle to a convolution function used in backward-prop
        :param conv_grad: handle to a convolution function used in pgrads
        :return:
        """
        
        if(rng == None):
            seed=[2015, 10, 1]
            self.rng = numpy.random.RandomState(seed)
        else:
            self.rng = rng
        
        self.num_out_feat_maps = num_out_feat_maps
        self.kernel_shape_x = kernel_shape[0]
        self.kernel_shape_y = kernel_shape[1]
        #Implement bias like kernels, where kernels are our weights
        
        #Make weights, use 1 to help broadcasting, wouldn't allow to use None and saves having to reshape later
        self.kernels = self.rng.uniform(-irange, irange,(1,self.num_out_feat_maps, self.kernel_shape_x, self.kernel_shape_y)) 
        
        self.bias = numpy.zeros(self.num_out_feat_maps)
        
        super(ConvLinear, self).__init__(rng=rng)


    def fprop(self, inputs):
        # Linear(conv_fwd)?
        #Remember bias
        
        return my1_conv2d(inputs, self.kernels) + self.bias[None,:,None,None]
        

    def bprop(self, h, igrads):
        '''
        Deltas - All weights which connect to that point * deltas of layer in front * linear derivative?
        ograds - same as other layers?
        '''
        raise NotImplementedError()
        
        
        return igrads, ograds
        

    def bprop_cost(self, h, igrads, cost):
        #No need to implement cost, as we won't ever use it as an output.
        raise NotImplementedError('ConvLinear.bprop_cost method not implemented')

    def pgrads(self, inputs, deltas, l1_weight=0, l2_weight=0):
        
        pgrads = numpy.zeros(shape=(numberOfInputFeatureMaps, numberOfOutputFeatureMaps, kernalSizeX, kernalSizeY))

        for b in range(0, imageBatchSize):
            for oFeatureMap in range(0, numberOfOutputFeatureMaps):
                for xloop in range(0, imageSizeX - kernalSizeX + 1, strideX):
                    for yloop in range(0, imageSizeY - kernalSizeY + 1, strideY):
                        singleDelta = deltas[b, oFeatureMap, xloop, yloop]
                        imageSlice = inputs[b, :, xloop:xloop + kernalSizeX, yloop:yloop + kernalSizeY]
                        pgrads[:, oFeatureMap, :, :] += singleDelta + imageSlice
        
        return pgrads

    def get_params(self):
        return self.kernels, self.bias

    def set_params(self, params):
        self.kernels = params[0]
        self.bias = params[1]

    def get_name(self):
        return 'convlinear'

#you can derive here particular non-linear implementations:
#class ConvSigmoid(ConvLinear):
# EXACTLY SAME BUT WRAP SIGMOID AROUND IT, DOUBLE CHECK IT GIVES AS EXPECTED!
#...


class ConvMaxPool2D(Layer):
    def __init__(self,
                 num_feat_maps,
                 conv_shape,
                 pool_shape=(2, 2),
                 pool_stride=(2, 2)):
        """

        :param conv_shape: tuple, a shape of the lower convolutional feature maps output
        :param pool_shape: tuple, a shape of pooling operator
        :param pool_stride: tuple, a strides for pooling operator
        :return:
        """

        super(ConvMaxPool2D, self).__init__(rng=None)
        raise NotImplementedError()

    
    def fprop(self, inputs):
        raise NotImplementedError()

        '''
            Should get max from 2x2 kernel, implement normally then try to use function above.
        '''
        
        
    def bprop(self, h, igrads):
        #Pass back the igrads multiplied by pseduo weight matrix.
        raise NotImplementedError()

    def get_params(self):
        return []

    def pgrads(self, inputs, deltas, **kwargs):
        return []

    def set_params(self, params):
        pass

    def get_name(self):
        return 'convmaxpool2d'