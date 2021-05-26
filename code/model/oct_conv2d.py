
"""
Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution
https://arxiv.org/abs/1904.05049
"""

from tensorflow.keras import layers  
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

class OctConv2D(layers.Layer):
    def __init__(self, filters_out, alpha_out, kernel=None, kernel_size=(3,3), strides=(1,1), 
                    padding="same", kernel_initializer='glorot_uniform',
                    kernel_regularizer=None, kernel_constraint=None,
                    **kwargs):
        """
        OctConv2D : Octave Convolution for image( rank 4 tensors)
        filters_out: # output channels for low + high
        alpha_out: Low channel ratio (alpha=0 -> High input only, alpha=1 -> Low input only)
        kernel : [kernel_size[0], kernelsize[1], in, out]
        kernel_size : 3x3 by default, padding : same by default
        """
        # assert alpha_in >= 0 and alpha_in <= 1
        assert 0 <= alpha_out <= 1
        assert filters_out > 0 and isinstance(filters_out, int)
        super().__init__(**kwargs)

        # self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.filters_out = filters_out
        # optional values
        self.kernel = kernel
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        # -> Low Channels 
        self.low_outchannels = int(self.filters_out * self.alpha_out)
        # -> High Channles
        self.high_outchannels = self.filters_out - self.low_outchannels
        
    def build(self, input_shape):
        """
        input_shape: matrix for input tensor, shape as [[...], [...]] or [...]
        """
        if len(input_shape) == 2 and len(input_shape[0]) == 4 and len(input_shape[1]) == 4:
            # Assertion for high inputs
            assert input_shape[0][1] // 2 >= self.kernel_size[0]
            assert input_shape[0][2] // 2 >= self.kernel_size[1]
            # Assertion for low inputs
            assert input_shape[0][1] // input_shape[1][1] == 2
            assert input_shape[0][2] // input_shape[1][2] == 2
            # channels last for TensorFlow
            assert K.image_data_format() == "channels_last"
            # input channels
            high_in = int(input_shape[0][3])
            low_in = int(input_shape[1][3])
            if self.kernel == None:
                # High -> High
                self.high_to_high_kernel = self.add_weight(name="high_to_high_kernel", 
                                            shape=(*self.kernel_size, high_in, self.high_outchannels),
                                            initializer=self.kernel_initializer,
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint)
                # High -> Low
                self.high_to_low_kernel  = self.add_weight(name="high_to_low_kernel", 
                                            shape=(*self.kernel_size, high_in, self.low_outchannels),
                                            initializer=self.kernel_initializer,
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint)
                # Low -> High
                self.low_to_high_kernel  = self.add_weight(name="low_to_high_kernel", 
                                            shape=(*self.kernel_size, low_in, self.high_outchannels),
                                            initializer=self.kernel_initializer,
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint)
                # Low -> Low
                self.low_to_low_kernel   = self.add_weight(name="low_to_low_kernel", 
                                            shape=(*self.kernel_size, low_in, self.low_outchannels),
                                            initializer=self.kernel_initializer,
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint)
            else:
                # bring in custom kernel
                assert self.filters_out == int(self.kernel.shape[3])
                self.high_to_high_kernel = self.kernel[:,:,:high_in, :self.high_outchannels]
                self.high_to_low_kernel = self.kernel[:,:,:high_in, self.high_outchannels:]
                self.low_to_high_kernel = self.kernel[:,:,high_in:, :self.high_outchannels]
                self.low_to_low_kernel = self.kernel[:,:,high_in:, self.high_outchannels:]

        elif len(input_shape) == 4:
            assert input_shape[1] // 2 >= self.kernel_size[0]
            assert K.image_data_format() == "channels_last"
            high_in = int(input_shape[3])
            if self.kernel == None:
                # High -> High
                self.high_to_high_kernel = self.add_weight(name="high_to_high_kernel", 
                                            shape=(*self.kernel_size, high_in, self.high_outchannels),
                                            initializer=self.kernel_initializer,
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint)
                # High -> Low
                self.high_to_low_kernel  = self.add_weight(name="high_to_low_kernel", 
                                            shape=(*self.kernel_size, high_in, self.low_outchannels),
                                            initializer=self.kernel_initializer,
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint)
            else:
                # bring in custom kernel
                self.high_to_high_kernel = self.kernel[:,:,:, :self.high_outchannels]
                self.high_to_low_kernel = self.kernel[:,:,:, self.high_outchannels:]
        super().build(input_shape)

    def call(self, inputs):
        # Input = [X^H, X^L]
        high_input, low_input = inputs if type(inputs) is list else (inputs, None)
        # High -> High conv
        high_to_high = K.conv2d(high_input, self.high_to_high_kernel,
                                strides=self.strides, padding=self.padding,
                                data_format="channels_last")
        # High -> Low conv
        high_to_low  = K.pool2d(high_input, (2,2), strides=(2,2), pool_mode="avg")
        high_to_low  = K.conv2d(high_to_low, self.high_to_low_kernel,
                                strides=self.strides, padding=self.padding,
                                data_format="channels_last")
        if low_input is not None:
            # Low -> High conv
            low_to_high  = K.conv2d(low_input, self.low_to_high_kernel,
                                    strides=self.strides, padding=self.padding,
                                    data_format="channels_last")
            low_to_high = K.repeat_elements(low_to_high, 2, axis=1) # Nearest Neighbor Upsampling
            low_to_high = K.repeat_elements(low_to_high, 2, axis=2)
            # Low -> Low conv
            low_to_low   = K.conv2d(low_input, self.low_to_low_kernel,
                                    strides=self.strides, padding=self.padding,
                                    data_format="channels_last")
            # Cross Add
            high_add = high_to_high + low_to_high
            low_add = high_to_low + low_to_low
        else:
            high_add = high_to_high
            low_add = high_to_low
        return [high_add, low_add]


    def compute_output_shape(self, input_shapes):
        high_in_shape, low_in_shape = input_shapes
        high_out_shape = (*high_in_shape[:3], self.high_channels)
        low_out_shape = (*low_in_shape[:3], self.low_channels)
        return [high_out_shape, low_out_shape]

    def get_config(self):
        base_config = super().get_config()
        out_config = {
            **base_config,
            "filters_out": self.filters_out,
            "alpha_out": self.alpha_out,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "kernel_constraint": self.kernel_constraint,            
        }
        return out_config