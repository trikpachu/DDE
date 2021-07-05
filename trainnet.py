# TODO: write nice dataloader which also incorporates random permutation of the input support dimensions 
#       to remove any dependency on the data orientation (e.g direction of monotionicyty)
#       apart from the dimension permutation this is also done now, but without a dataloader
import math
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import socket
import importlib
import sys
import time
#import matplotlib.pyplot as plt

class Net():
    def __init__(self, model_name='model_4',
                 num_point=16, max_epoch=250, batch_size=64, learning_rate=0.001,
                 decay_rate=0.7, decay_step=200000, log_dir='log',
                 step_count=24000, naming_parameter='', verbose=False, version=2):
        self.model_name = model_name
        self.num_point = num_point
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.base_learning_rate = learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.step_count = step_count
        self.naming_parameter = naming_parameter
        self.verbose = verbose
        self.version = version
        global tf
        if self.version == 1:
            import tensorflow.compat.v1 as tf
            import Models1 as Models
            tf.disable_eager_execution()
            self.MODEL = getattr(Models, self.model_name)(batch_size=self.batch_size, num_point=self.num_point)
        else:
            import tensorflow as tf
            import Models as Models
            self.MODEL = getattr(Models, self.model_name)()

    def exponential_decay(self, learning_rate, global_step, decay_steps, decay_rate, staircase=False):
        if staircase:
            return learning_rate * tf.pow(decay_rate, (global_step // decay_steps))
        else: 
            return learning_rate * tf.pow(decay_rate, (global_step / decay_steps))

    def get_learning_rate(self, batch):
        learning_rate = self.exponential_decay(
                        self.base_learning_rate,  
                        batch * self.batch_size,  
                        self.decay_step,          
                        self.decay_rate,          
                        staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.00001)  
        return learning_rate

    def get_loss_decay(self, batch):
        step = tf.divide(batch, self.step_count)
        loss_decay = tf.cast(tf.exp(-step), tf.float32)
        loss_decay = tf.maximum(loss_decay, tf.cast(0.001, tf.float32))
        return loss_decay

    def mean_squared_error_rel(self, y_true, y_pred):
        return np.average(((y_true - y_pred) / y_true) ** 2, axis=0)

    def mean_squared_error(self, y_true, y_pred):
        return np.average(((y_true - y_pred)) ** 2, axis=0)


    def train(self, distances, train_y, test_distances, test_y):
        distances = np.float32(distances)
        train_y = np.float32(train_y)
        test_distances = np.float32(test_distances)
        test_y = np.float32(test_y)

        trainings_run_identifier = f'{self.model_name}_{self.naming_parameter}_bs_{self.batch_size}_epochs_{self.max_epoch}_num_point_{self.num_point}_lr_{self.base_learning_rate}_ds_{self.decay_step}_dr_{self.decay_rate}'
        #plot_dir = f'figures/training/loss_curve_{trainings_run_identifier}.pdf'
        res_dir = f'result_files/test_results_{trainings_run_identifier}'

        #try: os.makedirs(os.path.dirname(plot_dir))
        #except: pass
        try: os.makedirs(os.path.dirname(res_dir))
        except: pass
        try: os.makedirs(f'log/{trainings_run_identifier}')
        except: pass

        @tf.function
        def train_step(batch, labels, loss_decay):
            with tf.GradientTape() as tape:
                prediction = Model(batch, training=True)
                loss = loss_object(prediction, labels, loss_decay=loss_decay)

            gradients = tape.gradient(loss, Model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, Model.trainable_variables))
            train_loss(loss)

        @tf.function
        def test_step(batch, labels):
            prediction = Model(batch, training=False)
            test_loss = loss_object(prediction, labels, loss_decay=loss_decay)
            return prediction, test_loss

        Model = self.MODEL.get_model(num_point=self.num_point)

        learning_rate_shedule = tf.keras.optimizers.schedules.ExponentialDecay(
                self.base_learning_rate, self.decay_step, self.decay_rate, staircase=True)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_shedule)
        #optimizer = tfa.optimizers.RectifiedAdam(lr=lr, total_steps=epochs+1, warmup_proportion=0.05, min_lr=1e-6)
        loss_object = self.MODEL.get_loss
        train_loss = tf.keras.metrics.Mean(name='train_loss')

        total_batch_count = 0
        history = {'train_loss':[], 'test_loss':[], 'train_idx':[], 'test_idx':[]}

        rest = np.shape(test_distances)[1] % self.batch_size

        min_loss = 1e8
        loss_epoch = 0
        stop_count = 0

        for epoch in range(self.max_epoch):
            if self.verbose: print(' epoch = ', epoch, flush=True)
            random_idxs = np.arange(0, len(train_y))
            np.random.shuffle(random_idxs)
            distances = distances[random_idxs]
            train_y = train_y[random_idxs]

            file_size = len(train_y)
            num_batches = file_size // self.batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx+1) * self.batch_size
                loss_decay = self.get_loss_decay(total_batch_count)
                train_step(distances[start_idx:end_idx], train_y[start_idx:end_idx], loss_decay)
                total_batch_count += 1
                history['train_loss'].append(train_loss.result().numpy())
                history['train_idx'].append(total_batch_count)

            epoch_test_loss = []
            
            for i in range(len(test_y)):  # loop over number of functions

                file_size = len(test_y[i])
                num_batches = np.int32(np.ceil(file_size / self.batch_size))

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * self.batch_size
                    end_idx = (batch_idx+1) * self.batch_size
                    if rest!=0 and batch_idx == (num_batches-1):
                        pred_val, test_loss = test_step(test_distances[i, start_idx:], tf.squeeze(test_y[i, start_idx:]))
                    else:
                        pred_val, test_loss = test_step(test_distances[i, start_idx:end_idx], tf.squeeze(test_y[i, start_idx:end_idx]))

                    if batch_idx == 0:
                        pred = pred_val
                    else:
                        pred = np.vstack((pred, pred_val))
                    epoch_test_loss.append(test_loss)
                if i == 0:
                    preds = pred
                else:
                    preds = np.dstack((preds, pred))

            epoch_test_loss = np.mean(epoch_test_loss)
            if self.verbose: print(' Validation Loss: ', epoch_test_loss)
            if epoch_test_loss < min_loss:
                min_loss = epoch_test_loss
                loss_epoch = epoch
                Model.save_weights(f'log/{trainings_run_identifier}/weights.ckpt')
                if self.verbose: print('saved model weights to log/{}/weights.ckpt'.format(trainings_run_identifier))
                stop_count = 0
            else:
                stop_count += 1

            if int(stop_count) == 10:
                break 

            preds = np.squeeze(preds)
            mse = []
            msre = []
            for i in range(len(test_y)):
                mse.append(self.mean_squared_error(test_y[i], preds[:, i]))
                msre.append(self.mean_squared_error_rel(test_y[i], preds[:, i]))

            mse = np.mean(mse)
            msre = np.mean(msre)

            history['test_loss'].append(mse)
            history['test_idx'].append(total_batch_count)

            results_log = open(res_dir, 'a')
            results_log.write(('         model= {:>9}  num_points= {:>7}  batch_size= {:>3}  '
                                    'num_point= {:>3}  epoch= {:>4} mean_of_mse= {:f}  '
                                    'mean_of_mse_rel= {:f} trainnet_nd\n'.format(
                                    self.model_name, str(len(train_y)), str(self.batch_size), str(self.num_point),
                                    str(epoch), mse, msre)))
            results_log.close()

        print('Training done.')
        print('The training was stopped after {} epochs'.format(loss_epoch+10))
        print('The best loss value was: ', min_loss)
        print('The best model weights are saved to log/{}/weights.ckpt'.format(trainings_run_identifier))
        #fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        #ax.set_title(plot_dir)
        #ax.plot(history['train_idx'], history['train_loss'], label='train_loss')
        #ax.plot(history['test_idx'], history['test_loss'], label='test_loss')
        #ax.legend()
        #fig.savefig(plot_dir)
        #plt.close(fig)

    def eval(self, test_distances, test_y, model_path=None):
        test_distances = np.float32(test_distances)
        test_y = np.float32(test_y)

        @tf.function
        def test_step(batch, labels):
            prediction = Model(batch, training=False)
            return prediction

        Model = self.MODEL.get_model(num_point=self.num_point)

        Model.load_weights(model_path)

        rest = np.shape(test_distances)[1] % self.batch_size
        if rest != 0:
            test_distances = np.pad(test_distances, ((0, 0), (0, self.batch_size-rest), (0, 0)), 'mean')
            test_y = np.pad(test_y, ((0, 0), (0, self.batch_size-rest)), 'mean')
        for i in range(len(test_y)):  # loop over number of functions
            file_size = len(test_y[i])
            num_batches = np.int32(np.ceil(file_size / self.batch_size))

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx+1) * self.batch_size
                pred_val = test_step(test_distances[i, start_idx:end_idx], np.squeeze(test_y[i, start_idx:end_idx]))

                if batch_idx == (num_batches - 1):
                    pred_val = pred_val[:rest]
                if batch_idx == 0:
                    pred = pred_val
                else:
                    pred = np.vstack((pred, pred_val))

            if i == 0:
                preds = pred
            else:
                preds = np.dstack((preds, pred))

        return np.squeeze(preds)

    def eval_one_epoch(self, sess, ops, test_distances, test_y):
        """ ops: dict mapping from string to tf ops """
        is_training = False

        total_seen = 0
        loss_sum = 0
        rest = np.shape(test_distances)[1] % self.batch_size

        if rest != 0:
            test_distances = np.pad(test_distances, ((0, 0), (0, self.batch_size-rest), (0, 0)), 'mean')
            test_y = np.pad(test_y, ((0, 0), (0, self.batch_size-rest)), 'mean')
        for i in range(len(test_y)):  # loop over number of functions

            file_size = len(test_y[i])
            num_batches = np.int32(np.ceil(file_size / self.batch_size))

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx+1) * self.batch_size
                feed_dict = {ops['distances_pl']: np.expand_dims(test_distances[i, start_idx:end_idx], -1),
                             ops['labels_pl']: np.squeeze(test_y[i, start_idx:end_idx]),
                             ops['is_training_pl']: is_training}

                summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                              ops['loss'], ops['pred']], feed_dict=feed_dict)
                if batch_idx == (num_batches - 1):
                    pred_val = pred_val[:rest]
                if batch_idx == 0:
                    pred = pred_val
                else:
                    pred = np.vstack((pred, pred_val))

                total_seen += self.batch_size
                loss_sum += (loss_val)

            if i == 0:
                preds = pred
            else:
                preds = np.dstack((preds, pred))

        return preds

    def load(self, model_path, gpu_index=0):

        with tf.device('/gpu:' + str(gpu_index)):
            distances_pl, labels_pl = self.MODEL.placeholder_inputs()
            is_training_pl = tf.placeholder(tf.bool, shape=())
            batch = tf.Variable(0)
            # simple model
            pred = self.MODEL.get_model(distances_pl, is_training_pl)
            loss = self.MODEL.get_loss(pred, labels_pl)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.gpu_options.visible_device_list = (str(gpu_index).encode('utf8').decode('utf8'))
        config.log_device_placement = True
        sess = tf.Session(config=config)
        merged = tf.summary.merge_all()
        # Restore variables from disk.
        saver.restore(sess, model_path)
        
        ops = {'distances_pl': distances_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'merged': merged,
               'step': batch}
    
        return sess, ops

    def eval1(self, sess, ops, test_distances, test_y):

        is_training = False

        pred = self.eval_one_epoch(sess, ops, test_distances, test_y)
        return np.squeeze(pred)