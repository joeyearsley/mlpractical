
# Machine Learning Practical (INFR11119),
# Pawel Swietojanski, University of Edinburgh


import numpy
import logging
from mlp.layers import Layer, max_and_argmax
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

def my1_conv2d(image, kernels, strides=(1, 1), pool=False):
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
    batch_size = image.shape[0]
    
    #Get the kernels size
    out = kernels.shape[1]
    
    '''
    
    Image dims - Used equation from stanford lectures: http://cs231n.github.io/convolutional-networks/
    if((InputSize-FilterSize+2Padding)/Stride+1) is valid int then carry on else throw and error, first calculate
    '''
    
    #Padding not implemented
    padding = 0
    
    #Get kernel sizes
    if(pool):
        kxdim = kernels.shape[0]
        kydim = kernels.shape[1]
    else:
        kxdim = kernels.shape[2]
        kydim = kernels.shape[3]
    
    #Can calculate here as this is all we are going to go to anyway.
    xdims = ((image.shape[2] - kxdim + (2*padding))/(strides[0])+1)
    ydims = ((image.shape[3] - kydim + (2*padding))/(strides[1])+1)
    
    #Do assertions to ensure passed in the correct type.
    assert type(xdims) is IntType,"Can't make feature map with x-stride: %r" % strides[0]
    assert type(ydims) is IntType,"Can't make feature map with y-stride: %r" % strides[1]
    
    
    
    #Get feature map size out
    num_out_feat_maps = kernels.shape[1]
    #Get input feature size
    num_inp_feat_maps = kernels.shape[0]
     
    #Create G matrix
    G = numpy.zeros(image.shape)
    
    #Create empty 4D tensor
    output =  numpy.zeros((batch_size,out,xdims,ydims))
    
    #For each image in batch
    for img in xrange(batch_size):
        #For each feature map (output map)
        for fm in xrange(num_out_feat_maps):
            #For each x-dim in output
            for x in xrange(0,xdims,strides[0]):
                #For each y-dim in output
                for y in xrange(0,ydims,strides[1]):
                    #Get image slice from entire image, corresponds to kernel size, accross all channels.
                    if(pool == True):
                        imgSlice = image[img, fm, x:x+kxdim, y:y+kydim]
                        maxi, maxInd = max_and_argmax(imgSlice)
                        output[img, fm, x/strides[0], y/strides[1]] = maxi
                        #Search the pool for all points corresponding to the max, fine for small filter
                        for xin in range(imgSlice.shape[0]):
                            for yin in range(imgSlice.shape[1]):
                                if imgSlice[xin,yin] == maxi:
                                    G[img,fm,x+xin, y+yin] = 1
                        
                    else:
                        imgSlice = image[img, :, x:x+kxdim, y:y+kydim]
                        #Get kernels accross all channels.
                        kernel = kernels[:, fm, :, :]
                        '''
                        Do the dot product to get the position.
                        '''
                        output[img, fm, x/strides[0], y/strides[1]] = numpy.dot(imgSlice.flatten(),kernel.flatten())
                
    return output,G

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
        
        self.num_inp_feat_maps = num_inp_feat_maps
        self.num_out_feat_maps = num_out_feat_maps
        self.kernel_shape_x = kernel_shape[0]
        self.kernel_shape_y = kernel_shape[1]
        #Implement bias like kernels, where kernels are our weights
        
        #Make weights, use 1 to help broadcasting, wouldn't allow to use None and saves having to reshape later
        self.kernels = self.rng.uniform(-irange, irange,(self.num_inp_feat_maps,self.num_out_feat_maps, self.kernel_shape_x, self.kernel_shape_y)) 
        
        self.bias = numpy.zeros(self.num_out_feat_maps)
        
        super(ConvLinear, self).__init__(rng=rng)


    def fprop(self, inputs):
        # Do usual convolution and add broadcasted bias
        retx,_ = my1_conv2d(inputs, self.kernels)
        
        for inx in xrange(0, self.num_out_feat_maps):
            retx[:,inx,:,:] += self.bias[inx]
        return  retx
        

    def bprop(self, h, igrads):
        '''
        Deltas - All weights which connect to that point * deltas of layer in front * linear derivative?
        ograds - same as other layers?
        
        
        Rotate kernel matrix 180 before passing into func
        Pad igrads before passing in, use zeros array
        '''
        
        #Pad with igrads with zeros either side.
        m = self.kernel_shape_x
        ig = numpy.pad(igrads, ((0,0),(0,0),(m-1,m-1),(m-1,m-1)), mode='constant', constant_values=0)
        
        tempKernels = numpy.zeros((self.num_out_feat_maps,self.num_inp_feat_maps,self.kernel_shape_x ,self.kernel_shape_y))
        tempKernels = self.kernels.swapaxes(0,1)
        print tempKernels.shape,'temp'
        #Rotate as expected along specific axes
        for n in range(0,self.num_out_feat_maps):
            for m in range(0,self.num_inp_feat_maps):
                tempKernels[n,m,:,:] = numpy.rot90(tempKernels[n,m,:,:], 2)
                
                
        #Swap axes as we are going backwards.
        ograds,_ = my1_conv2d(ig,tempKernels)#.swapaxes(0,1))
        
        return igrads, ograds
        

    def bprop_cost(self, h, igrads, cost):
        #No need to implement cost, as we won't ever use it as an output.
        raise NotImplementedError('ConvLinear.bprop_cost method not implemented')

    def pgrads(self, inputs, deltas, l1_weight=0, l2_weight=0):
        
        l2_W_penalty, l2_b_penalty = 0, 0
        if l2_weight > 0:
            l2_W_penalty = l2_weight*self.kernels
            l2_b_penalty = l2_weight*self.bias

        l1_W_penalty, l1_b_penalty = 0, 0
        if l1_weight > 0:
            l1_W_penalty = l1_weight*numpy.sign(self.kernels)
            l1_b_penalty = l1_weight*numpy.sign(self.bias)
            
        #input into conv
        grad_p,_ = my1_conv2d(inputs.swapaxes(0,1),deltas)
        
        grad_p += l2_W_penalty + l1_W_penalty
        
        #Bias sum all deltas
        grad_b = numpy.zeros(deltas.shape[1])
        for delt in xrange(deltas.shape[1]):
            grad_b[delt] = numpy.sum(deltas[:,delt,:,:].flatten())
         
        grad_b = grad_b + l2_b_penalty + l1_b_penalty
            
        return grad_p, grad_b

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

class ConvSigmoid(ConvLinear):
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

        super(ConvSigmoid, self).__init__(num_inp_feat_maps,
                 num_out_feat_maps,
                 image_shape=(28, 28),
                 kernel_shape=(5, 5),
                 stride=(1, 1),
                 irange=0.2,
                 rng=None,
                 )
    def fprop(self, inputs):
        #get the linear activations
        a = super(ConvSigmoid, self).fprop(inputs)
        #stabilise the exp() computation in case some values in
        #'a' get very negative. We limit both tails, however only
        #negative values may lead to numerical issues -- exp(-a)
        #clip() function does the following operation faster:
        # a[a < -30.] = -30,
        # a[a > 30.] = 30.
        numpy.clip(a, -30.0, 30.0, out=a)
        #Should all broadcast
        h = 1.0/(1 + numpy.exp(-a))
        return h
    
    def bprop(self, h, igrads):
        #Should broadcast
        dsigm = h * (1.0 - h)
        deltas = igrads * dsigm
        #As usual
        ___, ograds = super(ConvSigmoid, self).bprop(h=None, igrads=deltas)
        return deltas, ograds

    def get_name(self):
        return 'convsigmoid'

class ConvRelu(ConvLinear):
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

        super(ConvRelu, self).__init__(num_inp_feat_maps,
                 num_out_feat_maps,
                 image_shape=(28, 28),
                 kernel_shape=(5, 5),
                 stride=(1, 1),
                 irange=0.2,
                 rng=None,
                 )

    def fprop(self, inputs):
        #get the linear activations
        a = super(ConvRelu, self).fprop(inputs)
        h = numpy.clip(a, 0, 20.0)
        #h = numpy.maximum(a, 0)
        return h

    def bprop(self, h, igrads):
        deltas = (h > 0)*igrads
        ___, ograds = super(ConvRelu, self).bprop(h=None, igrads=deltas)
        return deltas, ograds

    def bprop_cost(self, h, igrads, cost):
        raise NotImplementedError('Relu.bprop_cost method not implemented '
                                      'for the %s cost' % cost.get_name())

    def get_name(self):
        return 'convrelu'

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
        
        self.stride = pool_stride
        self.pool_shape = pool_shape
        self.feat = num_feat_maps
        self.pool = numpy.ones(self.pool_shape)
        
        #Use max_and_argmax to find max pools.

    
    def fprop(self, inputs):
        '''
            Should get max from 2x2 kernel, implemented with normal conv to save code space
        '''
        out, self.G = my1_conv2d(inputs, self.pool, strides=self.stride, pool = True)
        return out
    
    def bprop(self, h, igrads):
        
        #First reshape the igrads to the same size as the 
        igrads = igrads.reshape(igrads.shape[0], h.shape[1], h.shape[2], h.shape[3])
        print igrads.shape
        
        #Make a copy!
        self.ograds = self.G.copy()
        
        for b in xrange(igrads.shape[0]):
            for f in xrange(h.shape[1]):
                for x in xrange(h.shape[2]):
                    for y in xrange(h.shape[3]):
                        #Multiple by stride to get right area
                        xs = x*self.stride[0]
                        ys = y*self.stride[1]
                        '''
                            Multiply by igrads as all these igrads affect this area
                            In generic, you would have to += with the G matrix * igrads, 
                            also maybe change multiply by stride to plus
                        '''
                        self.ograds[b,f,xs:xs+self.pool_shape[0], ys:ys+self.pool_shape[1]] *= igrads[b,f,x,y]
        
        return igrads, self.ograds
        
        

    def get_params(self):
        return []

    def pgrads(self, inputs, deltas, **kwargs):
        return []

    def set_params(self, params):
        pass

    def get_name(self):
        return 'convmaxpool2d'