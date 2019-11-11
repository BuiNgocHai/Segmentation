from keras.layers import *
from keras.models import Model
from keras import applications
from keras.optimizers import *
from keras.activations import *
from keras.callbacks import *
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed

def mean_iou(num_classes):
    def iou(y_true, y_pred):
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred, num_classes)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        return score
    return iou


def build_ptit(shape, num_classes, lr_init, lr_decay,  alpha=1.0, include_top=True, weights=None):
    mbl = applications.mobilenet.MobileNet(weights=None, include_top=False, input_shape=shape)
    x = mbl.output

    model_tmp =  Model(inputs = mbl.input, outputs = x)
    layer5, layer8, layer13 = model_tmp.get_layer('conv_pw_5_relu').output, model_tmp.get_layer('conv_pw_8_relu').output, model_tmp.get_layer('conv_pw_13_relu').output

    fcn14 = Conv2D(filters=2 , kernel_size=1, name='fcn14')(layer13)
    fcn15 = Conv2DTranspose(filters=layer8.get_shape().as_list()[-1] , kernel_size=4, strides=2, padding='same', name='fcn15')(fcn14)
    fcn15_skip_connected = Add(name="fcn15_plus_vgg_layer8")([fcn15, layer8])
    fcn16 = Conv2DTranspose(filters=layer5.get_shape().as_list()[-1], kernel_size=4, strides=2, padding='same', name="fcn16_conv2d")(fcn15_skip_connected)
    # Add skip connection
    fcn16_skip_connected = Add(name="fcn16_plus_vgg_layer5")([fcn16, layer5])
    # Upsample again
    fcn17 = Conv2DTranspose(filters=4, kernel_size=16, strides=(8, 8), padding='same', name="fcn17", activation="softmax")(fcn16_skip_connected)

    model = Model(inputs = mbl.input, outputs = fcn17)

    model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
                  loss=[focal_loss(alpha=.25, gamma=2)],
                  metrics=[mean_iou(num_classes=num_classes)])
    
    return model
