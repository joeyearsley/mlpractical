import numpy
cimport numpy

from layers import max_and_argmax

from types import *


DTYPE = numpy.float64
ctypedef numpy.float64_t DTYPE_t

DTYPE3 = numpy.float32
ctypedef numpy.float32_t DTYPE3_t

DTYPE2 = numpy.int
ctypedef numpy.int_t DTYPE2_t

def my1_conv2d(numpy.ndarray[DTYPE_t, ndim=4] image,numpy.ndarray[DTYPE_t, ndim=4] kernels, strides=(1, 1), pool=False):
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
    
    cdef int kxdim = kernels.shape[2]
    cdef int kydim = kernels.shape[3]
    #Get feature map size out
    cdef int num_out_feat_maps = kernels.shape[1]
    #Get input feature size
    cdef int num_inp_feat_maps = image.shape[1]
        
    
    assert kxdim <= image.shape[2], 'Kernel can be max the size of the image'
    assert kydim <= image.shape[3], 'Kernel can be max the size of the image'
    
    #Can calculate here as this is all we are going to go to anyway - xdims is how many times kernel can move...
    #These only tell us if we can use strides
    cdef int strides_pos_x = ((image.shape[2] - kxdim + (2*padding))/(strides[0])) +1
    cdef int strides_pos_y = ((image.shape[3] - kydim + (2*padding))/(strides[1])) +1
    
    #Do assertions to ensure passed in the correct type, strides pos also correspond to the output size.
    assert type(strides_pos_x) == DTYPE2,"Can't make feature map with x-stride: %r" % strides[0]
    assert type(strides_pos_y) == DTYPE2,"Can't make feature map with y-stride: %r" % strides[1]
    
    #Create G matrix
    cdef numpy.ndarray G = numpy.zeros((image.shape[0],image.shape[1],image.shape[2],image.shape[3]))
    
    #Actual number of dimensions to traverse, +1 to count for range function
    cdef int xdims = (image.shape[2] - kxdim) + 1
    cdef int ydims = (image.shape[3] - kydim) + 1
    
    #Create empty 4D tensor
    cdef numpy.ndarray output =  numpy.zeros((batch_size,out,strides_pos_x,strides_pos_y), dtype=DTYPE)
    cdef DTYPE3_t imgSlice
    cdef DTYPE_t kernel
    
    cdef numpy.ndarray imgSliceP
    
    cdef img,fm,x,y
    #For each image in batch
    for img in xrange(batch_size):
        for infm in xrange(num_inp_feat_maps):
            #For each feature map (output map)
            for fm in xrange(num_out_feat_maps):
            
                #For each x-dim in output
                '''
                    Striding is taken care of by going through all input x dimensions with a stride, then putting them into the
                    output file 
                '''
                for x in xrange(0,xdims,strides[0]):
                    #For each y-dim in output
                    for y in xrange(0,ydims,strides[1]):
                        #Get image slice from entire image, corresponds to kernel size, accross all channels.
                        if(pool == True):
                            imgSliceP = image[img, fm, x:x+kxdim, y:y+kydim]
                            maxi, maxInd = max_and_argmax(imgSliceP, keepdims_argmax=True)
                            #No need to div by strides
                            output[img, fm, x/strides[0], y/strides[1]] = maxi
                            #Add first corresponding max into G mat
                            G[img,fm,x+maxInd[0],y+maxInd[1]] = 1

                        else:
                            for xin in xrange(kxdim):
                                for yin in range(kydim):
                                    imgSlice = image[img, infm, x+xin, y+yin]
                                    #Get kernels accross all channels.
                                    kernel = kernels[infm, fm, xin, yin]

                                    '''
                                        Do the dot product to get the position.
                                        Divide by strides to get actual output position
                                    '''
                                    output[img, fm, x/strides[0], y/strides[1]] += imgSlice * kernel
                
    return output,G

