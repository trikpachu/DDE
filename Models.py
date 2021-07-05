import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers as l
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import utils as tf_util

@tf.function
def mse(pred, truth):
    return tf.reduce_mean(tf.square(tf.subtract(pred, truth)))

@tf.function
def rel_mse(pred, truth):
    return tf.reduce_mean(tf.square(tf.divide(tf.subtract(pred, truth), truth)))

class model_1():
    def __init__(self):
        pass

    def get_model(self, num_point):
        inputs = l.Input(shape=(num_point,))
        net = tf_util.fully_connected(inputs, 256, bn=True)
        net = tf_util.fully_connected(net, 128, bn=True)
        net = tf_util.fully_connected(net, 64, bn=True)
        net = tf_util.fully_connected(net, 32, bn=True)
        outputs = tf_util.fully_connected(net, 1, bn=False, activation_fn=tf.nn.relu)
        
        return Model(inputs=inputs, outputs=outputs)

    @tf.function
    def get_loss(self, pred, label, loss_decay=0):
        label = tf.expand_dims(label, -1)
        loss = tf.add(tf.multiply(mse(pred, label), loss_decay),
                    tf.multiply(rel_mse(pred, label), tf.subtract(tf.cast(1, tf.float32), loss_decay)))

        tf.summary.scalar('loss', loss)
        return loss

class model_2():
    def __init__(self):
        pass

    def get_model(self, num_point):
        inputs = l.Input(shape=(num_point,))

        net = tf_util.fully_connected(
            inputs, 128, bn=True)

        net = tf_util.fully_connected(
            net, 256, bn=True)

        net = tf_util.fully_connected(
            net, 512, bn=True)

        net = tf_util.fully_connected(
            net, 1024, bn=True)

        net = tf_util.fully_connected(
            net, 512, bn=True)

        net = tf_util.fully_connected(
            net, 256, bn=True)

        net = tf_util.fully_connected(
            net, 128, bn=True)

        net = tf_util.fully_connected(
            net, 64, bn=True)

        net = tf_util.fully_connected(
            net, 32, bn=True)

        outputs = tf_util.fully_connected(net, 1, bn=False, activation_fn=tf.nn.relu)

        return Model(inputs=inputs, outputs=outputs)

    def get_loss(self, pred, label, loss_decay=0):
        label = tf.expand_dims(label, -1)
        loss = tf.add(tf.multiply(mse(pred, label), loss_decay),
                    tf.multiply(rel_mse(pred, label), tf.subtract(tf.cast(1, tf.float32), loss_decay)))

        tf.summary.scalar('loss', loss)
        return loss

class model_3():
    def __init__(self):
        pass

    def get_model(self, num_point):
        inputs = l.Input(shape=(num_point,))

        net1 = tf_util.fully_connected(inputs, 256, bn=True)
        net = tf.concat((inputs, net1),1)
        net2 = tf_util.fully_connected(net, 256, bn=True)
        net = tf.concat((net1, net2),1)
        net3 = tf_util.fully_connected(net, 256, bn=True)
        net = tf.concat((net2, net3),1)
        net4 = tf_util.fully_connected(net, 256, bn=True)
        net = tf.concat((net3, net4),1)
        net5 = tf_util.fully_connected(net, 256, bn=True)

        net = tf.concat((net4, net5),1)
        net1 = tf_util.fully_connected(net, 128, bn=True)
        net = tf.concat((net5, net1),1)
        net2 = tf_util.fully_connected(net, 128, bn=True)
        net = tf.concat((net1, net2),1)
        net3 = tf_util.fully_connected(net, 128, bn=True)
        net = tf.concat((net2, net3),1)
        net4 = tf_util.fully_connected(net, 128, bn=True)
        net = tf.concat((net3, net4),1)
        net5 = tf_util.fully_connected(net, 128, bn=True)

        net = tf.concat((net4, net5),1)
        net1 = tf_util.fully_connected(net, 64, bn=True)
        net = tf.concat((net5, net1),1)
        net2 = tf_util.fully_connected(net, 64, bn=True)
        net = tf.concat((net1, net2),1)
        net3 = tf_util.fully_connected(net, 64, bn=True)
        net = tf.concat((net2, net3),1)
        net4 = tf_util.fully_connected(net, 64, bn=True)
        net = tf.concat((net3, net4),1)
        net5 = tf_util.fully_connected(net, 64, bn=True)

        net = tf.concat((net4, net5),1)
        net1 = tf_util.fully_connected(net, 32, bn=True)
        net = tf.concat((net5, net1),1)
        net2 = tf_util.fully_connected(net, 32, bn=True)
        net = tf.concat((net1, net2),1)
        net3 = tf_util.fully_connected(net, 32, bn=True)
        net = tf.concat((net2, net3),1)
        net4 = tf_util.fully_connected(net, 32, bn=True)
        net = tf.concat((net3, net4),1)
        net5 = tf_util.fully_connected(net, 32, bn=True)

        outputs = tf_util.fully_connected(net5, 1, bn=False, activation_fn=tf.nn.relu)

        return Model(inputs=inputs, outputs=outputs)

    def get_loss(self, pred, label, loss_decay=0):
        label = tf.expand_dims(label, -1)
        loss = tf.add(tf.multiply(mse(pred, label), loss_decay),
                    tf.multiply(rel_mse(pred, label), tf.subtract(tf.cast(1, tf.float32), loss_decay)))

        tf.summary.scalar('loss', loss)
        return loss

class model_4():
    def __init__(self):
        pass

    def get_model(self, num_point):
        inputs = l.Input(shape=(num_point,))

        net = tf_util.fully_connected(inputs, 128, bn=True)

        net = tf_util.fully_connected(net, 256, bn=True)

        net = tf_util.fully_connected(net, 512, bn=True)

        net = tf_util.fully_connected(net, 1024, bn=True)

        net = tf_util.fully_connected(net, 512, bn=True)

        net = tf_util.fully_connected(net, 256, bn=True)

        net = tf_util.fully_connected(net, 128, bn=True)

        net = tf_util.fully_connected(net, 64, bn=True)

        net = tf_util.fully_connected(net, 32, bn=True)

        outputs = tf_util.fully_connected(net, 1, bn=False, activation_fn=tf.nn.relu)

        return Model(inputs=inputs, outputs=outputs)

    @tf.function
    def get_loss(self, pred, label, loss_decay=0):
        label = tf.expand_dims(label, -1)
        loss = mse(pred, label)

        tf.summary.scalar('loss', loss)
        return loss

class model_5():
    def __init__(self):
        pass

    def get_model(self, num_point):
        inputs = l.Input(shape=(num_point,))

        net1 = tf_util.fully_connected(inputs, 256, bn=True)
        net = tf.concat((inputs, net1),1)
        net2 = tf_util.fully_connected(net, 256, bn=True)
        net = tf.concat((net1, net2),1)
        net3 = tf_util.fully_connected(net, 256, bn=True)
        net = tf.concat((net2, net3),1)
        net4 = tf_util.fully_connected(net, 256, bn=True)
        net = tf.concat((net3, net4),1)
        net5 = tf_util.fully_connected(net, 256, bn=True)

        net = tf.concat((net4, net5),1)
        net1 = tf_util.fully_connected(net, 128, bn=True)
        net = tf.concat((net5, net1),1)
        net2 = tf_util.fully_connected(net, 128, bn=True)
        net = tf.concat((net1, net2),1)
        net3 = tf_util.fully_connected(net, 128, bn=True)
        net = tf.concat((net2, net3),1)
        net4 = tf_util.fully_connected(net, 128, bn=True)
        net = tf.concat((net3, net4),1)
        net5 = tf_util.fully_connected(net, 128, bn=True)

        net = tf.concat((net4, net5),1)
        net1 = tf_util.fully_connected(net, 64, bn=True)
        net = tf.concat((net5, net1),1)
        net2 = tf_util.fully_connected(net, 64, bn=True)
        net = tf.concat((net1, net2),1)
        net3 = tf_util.fully_connected(net, 64, bn=True)
        net = tf.concat((net2, net3),1)
        net4 = tf_util.fully_connected(net, 64, bn=True)
        net = tf.concat((net3, net4),1)
        net5 = tf_util.fully_connected(net, 64, bn=True)

        net = tf.concat((net4, net5),1)
        net1 = tf_util.fully_connected(net, 32, bn=True)
        net = tf.concat((net5, net1),1)
        net2 = tf_util.fully_connected(net, 32, bn=True)
        net = tf.concat((net1, net2),1)
        net3 = tf_util.fully_connected(net, 32, bn=True)
        net = tf.concat((net2, net3),1)
        net4 = tf_util.fully_connected(net, 32, bn=True)
        net = tf.concat((net3, net4),1)
        net5 = tf_util.fully_connected(net, 32, bn=True)

        outputs = tf_util.fully_connected(net5, 1, bn=False, activation_fn=tf.nn.relu)

        return Model(inputs=inputs, outputs=outputs)

    def get_loss(self, pred, label, loss_decay=0):
        label = tf.expand_dims(label, -1)
        loss = mse(pred, label)

        tf.summary.scalar('loss', loss)
        return loss

class model_6():
    def __init__(self):
        pass

    def get_model(self, num_point):
        inputs = l.Input(shape=(num_point,))

        net = tf_util.fully_connected(inputs, 64, bn=True)

        net = tf_util.fully_connected(net, 128, bn=True)

        net = tf_util.fully_connected(net, 256, bn=True)

        net = tf_util.fully_connected(net, 256, bn=True)

        net = tf_util.fully_connected(net, 128, bn=True)

        net = tf_util.fully_connected(net, 64, bn=True)

        net = tf_util.fully_connected(net, 32, bn=True)

        outputs = tf_util.fully_connected(net, 1, bn=False, activation_fn=tf.nn.relu)
        
        return Model(inputs=inputs, outputs=outputs)

    def get_loss(self, pred, label, loss_decay=0):
        label = tf.expand_dims(label, -1)
        loss = mse(pred, label)

        tf.summary.scalar('loss', loss)
        return loss