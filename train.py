'''
TODO: Put Link of paper, when it is available
TODO: add option to generate data during training.
TODO: implement option for training continuation of saved model.
'''
import numpy as np
import pickle
from trainnet import Net
from sklearn.neighbors import NearestNeighbors
import copy

class trainer():
    '''
    This is an implementation of the training routines compatible to the new datageneration with function_generation_nd.
    This script is currently only supported for python 3.
    The script imports NEt from trainnet as the training operations and acts as a wrapper for data preprocessing steps.
    Note that this trains only  one model. To mimic the process described in the DDE paper, several models willl need to be trained, 
    out of which the best performing (on validation) is selected.
    
    Args:
        batch_size:         int, batch size for the neural net, default=512
        num_point:          int, number of drawn neighbors, i.e. kernel size of DDE, default=128
        model:              string, neural net model to use, located in Models.py, default = model_4
        max_epoch:          int, maximum number of epochs if early stopping did not trigger, default = 1000
        model_path:         string, trained model to load for continuation
        decay_step:         int, default=200000, decaystep of learning rate, default = 200000
        decay_rate:         float, decayrate of learning rate, default = 0.7
        learning_rate:      float, initial learning rate, default = 0.001
        continue_training:  bool, continue training the model in model_path?, default=False, NOT IMPLEMENTED YET
        renormalize:        bool, Renormalizing the data?, default=True. Data must be renormalized, only set to false if the data is already renormalized
        training_data:      array_like or string, training data. structure either [2, num_funcs, dim+1, num_points] or [num_funcs, dim+1, num_points]. 
                            In the former case it must be a list of training and validation data in that order. if a string is provided, the data will be loaded from that, expecting a pickle file with python3 encoding in binary (saved with 'wb').
                            If no validatio_data is provided and shape=[num_funcs, dim+1, num_points], the last quarter of the data will be used as validation data.
                            The last two dimensions may be swapped, in that case, dims_last needs to be provided.
        dims_last:          bool, indicates whether the data order is [num_funcs, dim+1, num_points] or [num_funcs, num_points, dim+1]                    
        validation_data:    array_like or string, training data. structure [num_funcs, dim+1, num_points].
                            If a string is provided, the data will be loaded from that, expecting a pickle file with python3 encoding in binary (saved with 'wb').
                            If no validatio_data is provided, it will be taken from training data.
                            The last two dimensions may be swapped, in that case, dims_last needs to be provided.
                            Same shape for trainig and validation data is expected.
        with_nn:            bool, set to True if the data already contains neighbours. (same expected for both training and validation data)
                            This is recommended for training of several similar models to speed up the process.
        name_add:           String, addition to the save_path of the trained model. otherwise it is determined by the model, training dataset dimensions, num_point and the network parameters. default=None
    '''

    def __init__(self, batch_size=512, num_point=128, model='model_4', max_epoch=1000, model_path=None, 
                 continue_training=False, decay_step=200000, decay_rate=0.7, learning_rate=0.001, renormalize=True, 
                 training_data=None, validation_data=None, dims_last=False, with_nn=False, name_add=None, verbose=False):
        self.verbose = verbose
        self.with_nn = with_nn
        self.num_point = num_point

        if training_data==None:
            raise ValueError('Please provide training data')
        else:
            if isinstance(training_data, str):
                if self.verbose: print('loading training data')
                try:
                    with open(training_data, 'rb') as f:
                        training_data = pickle.load(f)
                except:
                    raise ValueError('Could not load training data from {}'.format(training_data))
            if validation_data is not None and isinstance(validation_data, str):
                if self.verbose: print('loading validation data')
                try:
                    with open(validation_data, 'rb') as f:
                        validation_data = pickle.load(f)
                except:
                    raise ValueError('Could not load training data from {}'.format(validation_data))

            if self.shape_len(training_data)==4:
                self.training_data = training_data[0]
                self.validation_data = training_data[1]
            elif self.shape_len(training_data)==3:
                if validation_data is None:
                    self.training_data = training_data[:int(0.75*len(training_data))]
                    self.validation_data = training_data[int(0.75*len(training_data)):]
                else:
                    self.training_data = training_data
                    self.validation_data = validation_data
            else:
                raise ValueError('Expected training data of shape [2, num_funcs, num_points, dim+1] or [num_funcs, num_points, dim+1]')
        
        if verbose: print('Shape of data after loading:')
        if verbose: print('     Training_data:      ', np.shape(self.training_data))
        if verbose: print('     Validation_data:    ', np.shape(self.validation_data))
        self.training_data = np.array(self.training_data)
        self.validation_data = np.array(self.validation_data)
        if not dims_last:
            self.training_data = np.transpose(self.training_data, axes=(0,2,1))
            self.validation_data = np.transpose(self.validation_data, axes=(0,2,1))
        assert len(np.shape(self.training_data))==3, 'expected all functions to have the same size and dimensionality'
        
        self.num_funcs = np.shape(self.training_data)[0]
        self.size = np.shape(self.training_data)[1]
        self.dim = np.shape(self.training_data)[2] - 1
        if self.with_nn:
            self.dim = self.dim - self.num_point
        self.renormalize = renormalize
        if self.renormalize:
            if self.verbose: print('Renormalizing the data')
            self.training_data = self.renorm(self.training_data)
            self.validation_data = self.renorm(self.validation_data)

        #self.continue_training = continue_training # not yet implemented
        #self.model_path = model_path # only used for continue training

        self.learning_rate = learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.batch_size = batch_size
        self.model = model
        self.max_epoch = max_epoch
        self.naming_parameter = f'dim_{self.dim}_size_{self.size}_num_funcs_{self.num_funcs}'
        if name_add is not None:
            self.naming_parameter = '_'.join((self.naming_parameter, name_add))
        if self.dim == 1:
            self.step_count = 24000
        elif self.dim == 2:
            self.step_count = 80000
        else:
            self.step_count = 100000

    def shape_len(self, x):
        '''
        Gets the depth (number of dimensions) for lists, where sub-lists have different length, meaning that len(np.shape()) won't work.
        Example:
            a = np.arange(3)
            l = [[a,a], [a,a,a,a], [a]]
            print(shape_len(l))
            >>> 3
        '''
        sl = 0
        t = x
        instances = (int, float, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)
        while True:
            try:
                len(t)
                t = t[0]
                sl+=1
            except:
                if isinstance(t, instances):
                    return sl
                else:
                    print(t)
                    raise ValueError('unexpected shape of array or array contained values other than int or float types')

    def renorm(self, a):
        '''
        renormalizes the domain of every function in a to unit range [0,1]^dim. Expects a to have shape (num_funcs, num_points, dim+1) 
        '''
        for i in range(len(a)):
            vol = 1
            for d in range(self.dim):
                x_min = np.min(a[i, :, d])
                x_max = np.max(a[i, :, d])
                x_range = x_max - x_min
                a[i, :, d] = (a[i, :, d] - x_min) / x_range
                vol *= x_range
            a[i, :, self.dim] = a[i, :, self.dim] * vol
        return a

    def get_knn(self, data, n_neighbors):
        neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(data)
        return neigh

    def draw_knn(self, neigh, target, n_neighbors):
        return neigh.kneighbors(target, n_neighbors, return_distance=True)
    
    def get_knns(self, data):
        '''
        Data is expected to have the form [num_funcs, size, dim+1]
        '''

        data_ = []
        n_neighbors = self.num_point + 1
        for i in range(len(data)):
            neigh = self.get_knn(np.reshape(data[i, :, :self.dim], (-1,self.dim)), n_neighbors)
            distance, _ = self.draw_knn(neigh, np.reshape(data[i, :, :self.dim], (-1,self.dim)), n_neighbors)
            distances = (distance[:,1:],)
            for d in range(self.dim+1):
                distances += (np.expand_dims(data[i, :, d], -1),)
            distances = np.hstack(distances)
            data_.append(copy.copy(distances))

        return np.array(data_)

    def run(self):
        if self.with_nn:
            if self.verbose: print('data already contained neighbours.')
            training_distances = self.training_data
            validation_distances = self.validation_data
        else:
            if self.verbose: print('preprocessing data to draw neighbours for the complete set')
            training_distances = self.get_knns(self.training_data)
            validation_distances = self.get_knns(self.validation_data)
        training_distances = np.reshape(training_distances, (self.num_funcs*self.size, self.num_point+self.dim+1))
        idxs = np.arange(0, len(training_distances))
        np.random.shuffle(idxs)
        training_distances = training_distances[idxs]

        distances = training_distances[:, :self.num_point]
        y_dist = validation_distances[:, :, :self.num_point]
        train_y = training_distances[:, -1]
        test_y = validation_distances[:, :, -1]
        if self.verbose: print('starting training')
        Net(batch_size=self.batch_size, model_name=self.model, num_point=self.num_point, max_epoch=self.max_epoch,
            decay_step=self.decay_step, decay_rate=self.decay_rate, learning_rate=self.learning_rate,
            step_count=self.step_count, naming_parameter=self.naming_parameter, verbose=self.verbose).train(distances, train_y, y_dist, test_y)
        
        print('training done!!!', flush=True)
