import tensorflow.compat.v1 as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import utils1 as tf_util 

class model_1():
    def __init__(self, batch_size, num_point):
        self.batch_size = batch_size
        self.num_point = num_point

    def placeholder_inputs(self):
        distances_pl = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_point, 1))
        labels_pl = tf.placeholder(tf.float32, shape=(self.batch_size, ))
        return distances_pl, labels_pl


    def get_model(self, point_cloud, is_training, bn_decay=None):
        net = tf.squeeze(point_cloud)

        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc0', bn_decay=bn_decay)


        net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)


        net = tf_util.fully_connected(net, 64, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)


        net = tf_util.fully_connected(net, 32, bn=True, is_training=is_training, scope='fc3', bn_decay=bn_decay)

        net = tf.reshape(net, (self.batch_size, 32))
        net = tf_util.fully_connected(net, 1, bn=False, activation_fn=tf.nn.relu, is_training=is_training, scope='fc4', bn_decay=bn_decay)

        return net


    def mse(self, pred, truth):
        return tf.reduce_mean(tf.square(tf.subtract(pred, truth)))


    def rel_mse(self, pred, truth):
        return tf.reduce_mean(tf.square(tf.divide(tf.subtract(pred, truth), truth)))


    def get_loss(self, pred, label, loss_decay=0):
        label = tf.expand_dims(label, -1)
        loss = tf.add(tf.multiply(self.mse(pred, label), loss_decay),
                    tf.multiply(self.rel_mse(pred, label), tf.subtract(tf.cast(1, tf.float32), loss_decay)))

        tf.summary.scalar('loss', loss)
        return loss

class model_2():
    def __init__(self, batch_size, num_point):
        self.batch_size = batch_size
        self.num_point = num_point

    def placeholder_inputs(self):
        distances_pl = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_point, 1))
        labels_pl = tf.placeholder(tf.float32, shape=(self.batch_size, ))
        return distances_pl, labels_pl


    def get_model(self, point_cloud, is_training, bn_decay=None):
        net = tf.squeeze(point_cloud)

        net = tf_util.fully_connected(
            net, 128, bn=True, is_training=is_training, scope='k0', bn_decay=bn_decay)

        net = tf_util.fully_connected(
            net, 256, bn=True, is_training=is_training, scope='k1', bn_decay=bn_decay)

        net = tf_util.fully_connected(
            net, 512, bn=True, is_training=is_training, scope='k2', bn_decay=bn_decay)

        net = tf_util.fully_connected(
            net, 1024, bn=True, is_training=is_training, scope='k3', bn_decay=bn_decay)

        net = tf_util.fully_connected(
            net, 512, bn=True, is_training=is_training, scope='k4', bn_decay=bn_decay)

        net = tf_util.fully_connected(
            net, 256, bn=True, is_training=is_training, scope='fc0', bn_decay=bn_decay)

        net = tf_util.fully_connected(
            net, 128, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)

        net = tf_util.fully_connected(
            net, 64, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)

        net = tf_util.fully_connected(
            net, 32, bn=True, is_training=is_training, scope='fc3', bn_decay=bn_decay)

        net = tf.reshape(net, (self.batch_size, 32))
        net = tf_util.fully_connected(net, 1, bn=False, activation_fn=tf.nn.relu,
                                    is_training=is_training, scope='fc4', bn_decay=bn_decay)

        return net


    def mse(self, pred, truth):
        return tf.reduce_mean(tf.square(tf.subtract(pred, truth)))


    def rel_mse(self, pred, truth):
        return tf.reduce_mean(tf.square(tf.divide(tf.subtract(pred, truth), truth)))


    def get_loss(self, pred, label, loss_decay=0):
        label = tf.expand_dims(label, -1)
        loss = tf.add(tf.multiply(self.mse(pred, label), loss_decay),
                    tf.multiply(self.rel_mse(pred, label), tf.subtract(tf.cast(1, tf.float32), loss_decay)))

        tf.summary.scalar('loss', loss)
        return loss

class model_3():
    def __init__(self, batch_size, num_point):
        self.batch_size = batch_size
        self.num_point = num_point

    def placeholder_inputs(self):
        distances_pl = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_point, 1))
        labels_pl = tf.placeholder(tf.float32, shape=(self.batch_size, ))
        return distances_pl, labels_pl


    def get_model(self, point_cloud, is_training, bn_decay=None):
        net = tf.squeeze(point_cloud)

        net1 = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc0', bn_decay=bn_decay)
        net = tf.concat((net, net1),1)
        net2 = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc0.1', bn_decay=bn_decay)
        net = tf.concat((net1, net2),1)
        net3 = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc0.2', bn_decay=bn_decay)
        net = tf.concat((net2, net3),1)
        net4 = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc0.3', bn_decay=bn_decay)
        net = tf.concat((net3, net4),1)
        net5 = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc0.4', bn_decay=bn_decay)

        net = tf.concat((net4, net5),1)
        net1 = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        net = tf.concat((net5, net1),1)
        net2 = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='fc1.1', bn_decay=bn_decay)
        net = tf.concat((net1, net2),1)
        net3 = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='fc1.2', bn_decay=bn_decay)
        net = tf.concat((net2, net3),1)
        net4 = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='fc1.3', bn_decay=bn_decay)
        net = tf.concat((net3, net4),1)
        net5 = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='fc1.4', bn_decay=bn_decay)

        net = tf.concat((net4, net5),1)
        net1 = tf_util.fully_connected(net, 64, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        net = tf.concat((net5, net1),1)
        net2 = tf_util.fully_connected(net, 64, bn=True, is_training=is_training, scope='fc2.1', bn_decay=bn_decay)
        net = tf.concat((net1, net2),1)
        net3 = tf_util.fully_connected(net, 64, bn=True, is_training=is_training, scope='fc2.2', bn_decay=bn_decay)
        net = tf.concat((net2, net3),1)
        net4 = tf_util.fully_connected(net, 64, bn=True, is_training=is_training, scope='fc2.3', bn_decay=bn_decay)
        net = tf.concat((net3, net4),1)
        net5 = tf_util.fully_connected(net, 64, bn=True, is_training=is_training, scope='fc2.4', bn_decay=bn_decay)

        net = tf.concat((net4, net5),1)
        net1 = tf_util.fully_connected(net, 32, bn=True, is_training=is_training, scope='fc3', bn_decay=bn_decay)
        net = tf.concat((net5, net1),1)
        net2 = tf_util.fully_connected(net, 32, bn=True, is_training=is_training, scope='fc3.1', bn_decay=bn_decay)
        net = tf.concat((net1, net2),1)
        net3 = tf_util.fully_connected(net, 32, bn=True, is_training=is_training, scope='fc3.2', bn_decay=bn_decay)
        net = tf.concat((net2, net3),1)
        net4 = tf_util.fully_connected(net, 32, bn=True, is_training=is_training, scope='fc3.3', bn_decay=bn_decay)
        net = tf.concat((net3, net4),1)
        net5 = tf_util.fully_connected(net, 32, bn=True, is_training=is_training, scope='fc3.4', bn_decay=bn_decay)

        net = tf.reshape(net5, (self.batch_size, 32))
        net = tf_util.fully_connected(net, 1, bn=False, activation_fn=tf.nn.relu, is_training=is_training, scope='fc4', bn_decay=bn_decay)

        return net


    def mse(self, pred, truth):
        return tf.reduce_mean(tf.square(tf.subtract(pred, truth)))


    def rel_mse(self, pred, truth):
        return tf.reduce_mean(tf.square(tf.divide(tf.subtract(pred, truth), truth)))


    def get_loss(self, pred, label, loss_decay=0):
        label = tf.expand_dims(label, -1)
        loss = tf.add(tf.multiply(self.mse(pred, label), loss_decay),
                    tf.multiply(self.rel_mse(pred, label), tf.subtract(tf.cast(1, tf.float32), loss_decay)))

        tf.summary.scalar('loss', loss)
        return loss

class model_4():
    def __init__(self, batch_size, num_point):
        self.batch_size = batch_size
        self.num_point = num_point

    def placeholder_inputs(self):
        distances_pl = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_point, 1))
        labels_pl = tf.placeholder(tf.float32, shape=(self.batch_size, ))
        return distances_pl, labels_pl


    def get_model(self, point_cloud, is_training, bn_decay=None):
        net = tf.squeeze(point_cloud)

        net = tf_util.fully_connected(
            net, 128, bn=True, is_training=is_training, scope='k0', bn_decay=bn_decay)

        net = tf_util.fully_connected(
            net, 256, bn=True, is_training=is_training, scope='k1', bn_decay=bn_decay)

        net = tf_util.fully_connected(
            net, 512, bn=True, is_training=is_training, scope='k2', bn_decay=bn_decay)

        net = tf_util.fully_connected(
            net, 1024, bn=True, is_training=is_training, scope='k3', bn_decay=bn_decay)

        net = tf_util.fully_connected(
            net, 512, bn=True, is_training=is_training, scope='k4', bn_decay=bn_decay)

        net = tf_util.fully_connected(
            net, 256, bn=True, is_training=is_training, scope='fc0', bn_decay=bn_decay)

        net = tf_util.fully_connected(
            net, 128, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)

        net = tf_util.fully_connected(
            net, 64, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)

        net = tf_util.fully_connected(
            net, 32, bn=True, is_training=is_training, scope='fc3', bn_decay=bn_decay)

        net = tf.reshape(net, (self.batch_size, 32))
        net = tf_util.fully_connected(net, 1, bn=False, activation_fn=tf.nn.relu,
                                    is_training=is_training, scope='fc4', bn_decay=bn_decay)

        return net


    def mse(self, pred, truth):
        return tf.reduce_mean(tf.square(tf.subtract(pred, truth)))


    def rel_mse(self, pred, truth):
        return tf.reduce_mean(tf.square(tf.divide(tf.subtract(pred, truth), truth)))


    def get_loss(self, pred, label, loss_decay=0):
        label = tf.expand_dims(label, -1)
        loss = self.mse(pred, label)

        tf.summary.scalar('loss', loss)
        return loss

class model_5():
    def __init__(self, batch_size, num_point):
        self.batch_size = batch_size
        self.num_point = num_point

    def placeholder_inputs(self):
        distances_pl = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_point, 1))
        labels_pl = tf.placeholder(tf.float32, shape=(self.batch_size, ))
        return distances_pl, labels_pl


    def get_model(self, point_cloud, is_training, bn_decay=None):
        net = tf.squeeze(point_cloud)

        net1 = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc0', bn_decay=bn_decay)
        net = tf.concat((net, net1),1)
        net2 = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc0.1', bn_decay=bn_decay)
        net = tf.concat((net1, net2),1)
        net3 = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc0.2', bn_decay=bn_decay)
        net = tf.concat((net2, net3),1)
        net4 = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc0.3', bn_decay=bn_decay)
        net = tf.concat((net3, net4),1)
        net5 = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc0.4', bn_decay=bn_decay)

        net = tf.concat((net4, net5),1)
        net1 = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        net = tf.concat((net5, net1),1)
        net2 = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='fc1.1', bn_decay=bn_decay)
        net = tf.concat((net1, net2),1)
        net3 = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='fc1.2', bn_decay=bn_decay)
        net = tf.concat((net2, net3),1)
        net4 = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='fc1.3', bn_decay=bn_decay)
        net = tf.concat((net3, net4),1)
        net5 = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='fc1.4', bn_decay=bn_decay)

        net = tf.concat((net4, net5),1)
        net1 = tf_util.fully_connected(net, 64, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        net = tf.concat((net5, net1),1)
        net2 = tf_util.fully_connected(net, 64, bn=True, is_training=is_training, scope='fc2.1', bn_decay=bn_decay)
        net = tf.concat((net1, net2),1)
        net3 = tf_util.fully_connected(net, 64, bn=True, is_training=is_training, scope='fc2.2', bn_decay=bn_decay)
        net = tf.concat((net2, net3),1)
        net4 = tf_util.fully_connected(net, 64, bn=True, is_training=is_training, scope='fc2.3', bn_decay=bn_decay)
        net = tf.concat((net3, net4),1)
        net5 = tf_util.fully_connected(net, 64, bn=True, is_training=is_training, scope='fc2.4', bn_decay=bn_decay)

        net = tf.concat((net4, net5),1)
        net1 = tf_util.fully_connected(net, 32, bn=True, is_training=is_training, scope='fc3', bn_decay=bn_decay)
        net = tf.concat((net5, net1),1)
        net2 = tf_util.fully_connected(net, 32, bn=True, is_training=is_training, scope='fc3.1', bn_decay=bn_decay)
        net = tf.concat((net1, net2),1)
        net3 = tf_util.fully_connected(net, 32, bn=True, is_training=is_training, scope='fc3.2', bn_decay=bn_decay)
        net = tf.concat((net2, net3),1)
        net4 = tf_util.fully_connected(net, 32, bn=True, is_training=is_training, scope='fc3.3', bn_decay=bn_decay)
        net = tf.concat((net3, net4),1)
        net5 = tf_util.fully_connected(net, 32, bn=True, is_training=is_training, scope='fc3.4', bn_decay=bn_decay)

        net = tf.reshape(net5, (self.batch_size, 32))
        net = tf_util.fully_connected(net, 1, bn=False, activation_fn=tf.nn.relu, is_training=is_training, scope='fc4', bn_decay=bn_decay)

        return net


    def mse(self, pred, truth):
        return tf.reduce_mean(tf.square(tf.subtract(pred, truth)))


    def rel_mse(self, pred, truth):
        return tf.reduce_mean(tf.square(tf.divide(tf.subtract(pred, truth), truth)))


    def get_loss(self, pred, label, loss_decay=0):
        label = tf.expand_dims(label, -1)
        loss = self.mse(pred, label)

        tf.summary.scalar('loss', loss)
        return loss

class model_6():
    def __init__(self, batch_size, num_point):
        self.batch_size = batch_size
        self.num_point = num_point

    def placeholder_inputs(self):
        distances_pl = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_point, 1))
        labels_pl = tf.placeholder(tf.float32, shape=(self.batch_size, ))
        return distances_pl, labels_pl


    def get_model(self, point_cloud, is_training, bn_decay=None):
        net = tf.squeeze(point_cloud)

        net = tf_util.fully_connected(net, 64, bn=True, is_training=is_training, scope='k0', bn_decay=bn_decay)

        net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='k1', bn_decay=bn_decay)

        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='k2', bn_decay=bn_decay)

        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc0', bn_decay=bn_decay)

        net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)

        net = tf_util.fully_connected(net, 64, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)

        net = tf_util.fully_connected(net, 32, bn=True, is_training=is_training, scope='fc3', bn_decay=bn_decay)

        net = tf.reshape(net, (self.batch_size, 32))
        net = tf_util.fully_connected(net, 1, bn=False, activation_fn=tf.nn.relu, is_training=is_training, scope='fc4', bn_decay=bn_decay)
        return net


    def mse(self, pred, truth):
        return tf.reduce_mean(tf.square(tf.subtract(pred, truth)))


    def rel_mse(self, pred, truth):
        return tf.reduce_mean(tf.square(tf.divide(tf.subtract(pred, truth), truth)))


    def get_loss(self, pred, label, loss_decay=0):
        label = tf.expand_dims(label, -1)
        loss = self.mse(pred, label)

        tf.summary.scalar('loss', loss)
        return loss