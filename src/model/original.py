from keras.layers import Input
from keras.layers.core import Lambda, Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, UpSampling2D, MaxPooling2D
from keras.layers.merge import Add, Concatenate
from keras.layers.noise import GaussianNoise
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
from keras.models import Model
from keras import backend as K

import tensorflow as tf

def generate_model(weight_decay=0.0005, verbose=False, input_shape=(160, 60, 3)):
    def upsample_neighbor_function(input_x):
        input_x_pad = K.spatial_2d_padding(input_x, padding=((2,2),(2,2)))
        x_length = K.int_shape(input_x)[1]
        y_length = K.int_shape(input_x)[2]
        output_x_list = []
        output_y_list = []
        for i_x in range(2, x_length + 2):
            for i_y in range(2, y_length + 2):
                output_y_list.append(input_x_pad[:,i_x-2:i_x+3,i_y-2:i_y+3,:])
            output_x_list.append(K.concatenate(output_y_list, axis=2))
            output_y_list = []
        return K.concatenate(output_x_list, axis=1)

    def buildNetwork(weight_decay):
        max_pooling = MaxPooling2D()

        x1_input = Input(shape=(160, 60, 3))
        x2_input = Input(shape=(160, 60, 3))

        share_conv_1 = Conv2D(20, 5, kernel_regularizer=l2(weight_decay), activation="relu")
        x1 = share_conv_1(x1_input)
        x2 = share_conv_1(x2_input)
        x1 = max_pooling(x1)
        x2 = max_pooling(x2)

        x1 = GaussianNoise(0.1) (x1)
        x2 = GaussianNoise(0.1) (x2)

        share_conv_2 = Conv2D(25, 5, kernel_regularizer=l2(weight_decay), activation="relu")
        x1 = share_conv_2(x1)
        x2 = share_conv_2(x2)
        x1 = max_pooling(x1)
        x2 = max_pooling(x2)

        x1 = Dropout(0.4)(x1)
        x2 = Dropout(0.4)(x2)

        upsample_same = UpSampling2D(size=(5, 5))
        x1_up = upsample_same(x1)
        x2_up = upsample_same(x2)
        upsample_neighbor = Lambda(upsample_neighbor_function)
        x1_nn = upsample_neighbor(x1)
        x2_nn = upsample_neighbor(x2)
        negative = Lambda(lambda x: -x)
        x1_nn = negative(x1_nn)
        x2_nn = negative(x2_nn)
        x1 = Add()([x1_up, x2_nn])
        x2 = Add()([x2_up, x1_nn])

        conv_3_1 = Conv2D(25, 5, strides=(5, 5), kernel_regularizer=l2(weight_decay), activation="relu")
        conv_3_2 = Conv2D(25, 5, strides=(5, 5), kernel_regularizer=l2(weight_decay), activation="relu")
        x1 = conv_3_1(x1)
        x2 = conv_3_2(x2)

        x1 = Dropout(0.4)(x1)
        x2 = Dropout(0.4)(x2)

        conv_4_1 = Conv2D(25, 3, kernel_regularizer=l2(weight_decay), activation="relu")
        conv_4_2 = Conv2D(25, 3, kernel_regularizer=l2(weight_decay), activation="relu")
        x1 = conv_4_1(x1)
        x2 = conv_4_2(x2)
        x1 = max_pooling(x1)
        x2 = max_pooling(x2)

        x1 = Dropout(0.4)(x1)
        x2 = Dropout(0.4)(x2)

        y = Concatenate()([x1, x2])
        y = Flatten()(y)
        y = Dense(500, kernel_regularizer=l2(weight_decay), activation='relu')(y)
        y = Dense(2, kernel_regularizer=l2(weight_decay), activation='softmax')(y)

        return Model(inputs=[x1_input, x2_input], outputs=[y])

    class model_SGD(SGD):
        "Redefinition of the original SGD."
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def get_updates(self, params, constraints, loss):
            grads = self.get_gradients(loss, params)
            self.updates = []

            lr = self.lr
            if self.initial_decay > 0:
                lr *= (1. / (1. + self.decay * self.iterations)) ** 0.75
                self.updates.append(K.update_add(self.iterations, 1))

            # momentum
            shapes = [K.get_variable_shape(p) for p in params]
            moments = [K.zeros(shape) for shape in shapes]
            self.weights = [self.iterations] + moments
            for p, g, m in zip(params, grads, moments):
                v = self.momentum * m - lr * g  # velocity
                self.updates.append(K.update(m, v))

                if self.nesterov:
                    new_p = p + self.momentum * v - lr * g
                else:
                    new_p = p + v

                # apply constraints
                if p in constraints:
                    c = constraints[p]
                    new_p = c(new_p)

                self.updates.append(K.update(p, new_p))
            return self.updates

    model = buildNetwork(weight_decay)

    model_loss = "categorical_crossentropy"
    model_opt = model_SGD()

    model.compile(optimizer=model_opt,
                  loss=model_loss)

    return model
