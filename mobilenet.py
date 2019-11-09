from keras.optimizers import *
from keras.models import Model
from keras.layers import *
from keras.activations import *
from keras.callbacks import *
from keras import backend as K

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def get_conv_block(tensor, channels, strides, alpha=1.0, name=''):
    channels = int(channels * alpha)

    x = Conv2D(channels,
               kernel_size=(1, 1),
               strides=strides,
               use_bias=False,
               padding='same',
               name='{}_conv'.format(name))(tensor)
    x = BatchNormalization(name='{}_bn'.format(name))(x)
    x = Activation('relu', name='{}_act'.format(name))(x)
    return x


def get_dw_sep_block(tensor, channels, strides, alpha=1.0, name=''):
    """Depthwise separable conv: A Depthwise conv followed by a Pointwise conv."""
    channels = int(channels * alpha)

    # Depthwise
    x = DepthwiseConv2D(kernel_size=(3, 3),
                        strides=strides,
                        use_bias=False,
                        padding='same',
                        name='{}_dw'.format(name))(tensor)
    x = BatchNormalization(name='{}_bn1'.format(name))(x)
    x = Activation('relu', name='{}_act1'.format(name))(x)

    # Pointwise
    x = Conv2D(channels,
               kernel_size=(1, 1),
               strides=(1, 1),
               use_bias=False,
               padding='same',
               name='{}_pw'.format(name))(x)
    x = BatchNormalization(name='{}_bn2'.format(name))(x)
    x = Activation('relu', name='{}_act2'.format(name))(x)
    return x

def get_transpose_block(tensor, channels, strides = (2,2), alpha=1.0, name=''):
	channels = int(channels * alpha)
	x = Conv2DTranspose(channels, kernel_size=(2, 2), strides=strides)(tensor)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	return x 		

def MobileNet(shape, num_classes, lr_init, lr_decay,  alpha=1.0, include_top=True, weights=None):
    x_in = Input(shape=shape)

    x = get_conv_block(x_in, 64, (1,1), alpha=alpha, name='initial')
    x = get_dw_sep_block(x, 64, (1,1),  name = 'block1')
    x = MaxPooling2D(strides = (2,2), padding ='same')(x)

    block_out1 = x 

    x = get_dw_sep_block(x, 128, (1,1), name = 'block2')
    x = get_dw_sep_block(x, 128, (1,1), name = 'block3')
    x = MaxPooling2D(strides = (2,2), padding ='same')(x)
    block_out2 = x 

    x = get_dw_sep_block(x, 256, (1,1), name = 'block4')
    x = get_dw_sep_block(x, 256, (1,1), name = 'block5')
    x = get_dw_sep_block(x, 256, (1,1), name = 'block6')
    x = MaxPooling2D(strides = (2,2), padding ='same')(x)
    block_out3 = x

    x = get_dw_sep_block(x, 512, (1,1), name = 'block7')
    x = get_dw_sep_block(x, 512, (1,1), name = 'block8')
    x = get_dw_sep_block(x, 512, (1,1), name = 'block9')
    x = MaxPooling2D(strides = (2,2), padding ='same')(x)
    block_out4 = x

    x = get_dw_sep_block(x, 512, (1,1), name = 'block10')
    x = get_dw_sep_block(x, 512, (1,1), name = 'block11')
    x = get_dw_sep_block(x, 512, (1,1), name = 'block12')
    #x = MaxPooling2D(strides = (2,2), padding ='same')(x)

    #up
    #up1
    x = get_transpose_block(x, 512)
    x = get_dw_sep_block(x, 512, (1,1), name = 'block13')
    x = get_dw_sep_block(x, 512, (1,1), name = 'block14')
    x = get_dw_sep_block(x, 512, (1,1), name = 'block15')
    #x = concatenate([x, block_out4])
    

    #up2

    x = get_transpose_block(x, 512)
    x = get_dw_sep_block(x, 512, (1,1), name = 'block16')
    x = get_dw_sep_block(x, 512, (1,1), name = 'block17')
    x = get_dw_sep_block(x, 256, (1,1), name = 'block18')
    block_out3 = ZeroPadding2D()(block_out3)
    #x = concatenate([x, block_out3])

    #up3

    x = get_transpose_block(x, 256)
    x = get_dw_sep_block(x, 256, (1,1), name = 'block19')
    x = get_dw_sep_block(x, 256, (1,1), name = 'block20')
    x = get_dw_sep_block(x, 128, (1,1), name = 'block21')
    block_out2 = ZeroPadding2D()(block_out2)
    #x = concatenate([x, block_out2])

    #up4

    x = get_transpose_block(x, 128)
    x = get_dw_sep_block(x, 128, (1,1), name = 'block22')
    x = get_dw_sep_block(x, 64, (1,1), name = 'block23')
    block_out1 = ZeroPadding2D()(block_out1)
    #x = concatenate([x, block_out1])

    #last
   # x = get_transpose_block(x, 64)
    x = get_dw_sep_block(x, 64, (1,1), name = 'block24')
    x = get_dw_sep_block(x, 64, (1,1), name = 'block25')

    x  = Conv2D(num_classes, (1, 1), activation='softmax', padding='same')(x)
    model = Model(x_in, x)
    model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])
    return model
