'''
TODO: add option to allow for dims_last.
TODO: allow for functions containig pdf(x) as in train
TODO: allow for different data structures as in train (needs looping as size differs betweeen functions, thus no np.array operations directly possible.)
TODO: add option for grid prediction. (Currently possible if grid is passed to estimate_at)
TODO: Retrain models for 1D to 3D to get rid of different tf versions in eval.
'''
import numpy as np
import pickle
from deep_density_estimation.trainnet import Net
from sklearn.neighbors import NearestNeighbors
import copy
import os
from scipy.interpolate import UnivariateSpline
from importlib.resources import files, as_file

class estimator():
    '''
    This is an implementation of the estimation routines.
    This script is currently only supported for python 3.
    The script imports NEt from trainnet as the training operations and acts as a wrapper for data preprocessing steps.
    
    Args:
        batch_size:         int, batch size for the neural net, default=512
        num_point:          int, number of drawn neighbors, i.e. kernel size of DDE, default=128
        model:              string, neural net model to use, located in Models.py, default = model_4. Model provided are all model_4.
        model_path:         string, trained model. Must be set for custom trained models.
        renormalize:        bool, Renormalizing the data?, default=True. Data must be renormalized, only set to false if the data is already renormalized
        dist:               array_like or string, test data structure [num_funcs, dim, num_points]. 
                            if a string is provided, the data will be loaded from that, expecting a pickle file with python3 encoding in binary (saved with 'wb').
                            dim (and nn if provided) is expected to be constant over all funstions. num_points may vary.
                            While the first dimesnion maybe omitted if only one function is passed, the second dimension must be apparent, even for dim==1.
                            only if bot num_funcs and dim are 1, dim can be omitted.
        with_nn:            bool, set to True if the data already contains neighbours.
                            This is recommended for testing of several similar models to speed up the process.
        training_size:      int, size of the training samples. Must be provided for custom trained models if it is not 5000. 
                            Otherwise it will be inferred from dimensionality (1000 for dim==1, 5000 else.). Default=None
        smoothing:          bool, if True smooth the 1D estimates using univariate splines. Default=True
        estimate_at:        array-like, floats. If provided, the estimation based on dist if conducted at these positions. expects shape (num_funcs, dim, size).
                            num_funcs and dim must be same as in dist. Default=None

    Returns: a list containing the estimates per function. [num_funcs, size]
    '''

    def __init__(self, batch_size=512, num_point=128, model='model_4', model_path=None, training_size=None,
                 renormalize=True, dist=None, estimate_at=None, with_nn=False, smoothing=True, verbose=False):
        self.verbose = verbose
        self.with_nn = with_nn
        self.num_point = num_point
        self.smoothing = smoothing
        if dist is None:
            raise ValueError('Please provide test data')
        else:
            if isinstance(dist, str):
                if self.verbose: print('loading test data')
                try:
                    with open(dist, 'rb') as f:
                        dist = pickle.load(f)
                except:
                    raise ValueError('Could not load test data from {}'.format(dist))

            if self.shape_len(dist)==1:
                self.dist = np.expand_dims(np.expand_dims(dist, 0), 0)
            elif self.shape_len(dist)==2:
                self.dist = np.expand_dims(dist, 0)
            elif self.shape_len(dist)==3:
                self.dist = dist
            else:
                raise ValueError('Expected test data of structure [num_funcs, num_points, dim+1] or [num_funcs, dim+1, num_points]')

        if estimate_at is not None:
            if isinstance(estimate_at, str):
                if self.verbose: print('loading estimation points')
                try:
                    with open(dist, 'rb') as f:
                        estimate_at = pickle.load(f)
                except:
                    raise ValueError('Could not load estimation points from {}'.format(estimate_at))

            if self.shape_len(estimate_at)==1:
                self.estimate_at = np.expand_dims(np.expand_dims(estimate_at, 0), 0)
            elif self.shape_len(estimate_at)==2:
                self.estimate_at = np.expand_dims(estimate_at, 0)
            elif self.shape_len(estimate_at)==3:
                self.estimate_at = estimate_at
            else:
                raise ValueError('Expected estimation points of structure [num_funcs, num_points, dim+1] or [num_funcs, dim+1, num_points]')
        else:
            self.estimate_at = estimate_at

        self.num_funcs = len(self.dist)
        self.dim = len(self.dist[0])
        if self.with_nn:
            self.dim = self.dim - self.num_point
        self.renormalize = renormalize
        if self.renormalize:
            if self.verbose: print('Renormalizing the data')
            self.dist, self.dist_volumes = self.renorm(self.dist)
            if self.estimate_at is not None:
                self.estimate_at, self.estimate_volumes = self.renorm(self.estimate_at)
            else:
                self.estimate_volumes = self.dist_volumes

        if self.estimate_at is None:
            self.estimate_at = self.dist
        self.model_path = model_path
        self.version = 2
        self.model = model
        
        if self.model_path is None:
            if self.num_point != 128:
                print('Only saved models for num_point = 128 available. Setting num_point = 128')
                self.num_point = 128
            if self.dim == 2:
                self.model = 'model_5'
            elif self.model != 'model_4':
                print('Only saved models for model = model_4 available. Setting model = model_4')
            if self.dim <= 3:
                self.version = 1
            #self.model_path = 'trained_states/{}d/{}d_{}.ckpt'.format(self.dim, self.dim, self.num_point)
            model_resources = files('deep_density_estimation')
            model_path_manager = as_file(model_resources / 'trained_states' / '{}d'.format(self.dim) / '{}d_{}.ckpt'.format(self.dim, self.num_point))
            with model_path_manager as mpm:
                self.model_path = str(mpm)
            print(self.model_path)
            assert os.path.exists(self.model_path+'.index'), 'could not find model {}'.format(self.model_path)
        
        self.batch_size = batch_size
        

        self.training_size = training_size
        if self.training_size is None:
            if self.dim == 1:
                self.training_size = 1000
            else:
                self.training_size = 5000

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
        renorm_list = []
        volumes = []
        for i in range(len(a)):
            vol = 1
            function = a[i]
            for d in range(self.dim):
                x_min = np.min(function[d])
                x_max = np.max(function[d])
                x_range = x_max - x_min
                function[d] = (function[d] - x_min) / x_range
                vol *= x_range
            volumes.append(vol)
            #function[self.dim] = function[self.dim] * vol
            renorm_list.append(copy.copy(function))
        return renorm_list, volumes

    def get_knn(self, data, n_neighbors):
        neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(data)
        return neigh

    def draw_knn(self, neigh, target, n_neighbors):
        return neigh.kneighbors(target, n_neighbors, return_distance=True)
    
    def get_knns(self, data, estimation):
        '''
        Data is expected to have the form [num_funcs, size, dim+1]
        '''
        if estimation is not None:
            estimation_points = estimation
        else:
            estimation_points = data
        data_ = []
        n_neighbors = self.num_point + 1
        for i in range(len(data)):
            function = np.transpose(np.array(data[i]), axes=(1,0))
            estim = np.transpose(np.array(estimation_points[i]), axes=(1,0))
            neigh = self.get_knn(np.reshape(function[:, :self.dim], (-1,self.dim)), n_neighbors)
            distance, _ = self.draw_knn(neigh, np.reshape(estim[:, :self.dim], (-1, self.dim)), n_neighbors)
            distances = distance[:, 1:]
            data_.append(copy.copy(distances))
        return np.array(data_)

    def run(self):
        net_preds = []

        if self.verbose: print('Starting Estimation')

        network = Net(batch_size=self.batch_size, model_name=self.model, num_point=self.num_point, verbose=self.verbose, version=self.version)
        if self.version == 1:
            sess, ops = network.load(model_path=self.model_path)

        if self.with_nn:
            if self.verbose: print('data already contained neighbours.')
            if self.estimate_at is not None:
                estimation_points = self.estimate_at
            else:
                estimation_points = self.dist
        else:
            if self.verbose: print('preprocessing data to draw neighbours for the complete set')
            estimation_points = self.get_knns(self.dist, self.estimate_at)

        for i in range(self.num_funcs):
            print(np.shape(estimation_points[i]))
            distances = np.array(estimation_points[i])
            size = len(distances)
            test_y = np.expand_dims(np.ones(size), 0)
            print(size, self.training_size, self.dim)
            distances = distances * np.power(float(size) / float(self.training_size), 1.0 / self.dim)
            distances = np.expand_dims(distances, 0)
            if self.version == 1:
                current_pred = np.squeeze(network.eval1(sess, ops, distances, test_y))
            else:
                current_pred = np.squeeze(network.eval(distances, test_y, model_path=self.model_path))
            if self.smoothing and self.dim == 1:
                sort_idx = np.argsort(np.squeeze(self.estimate_at[i]))
                x = np.squeeze(copy.copy(self.estimate_at[i]))[sort_idx]
                y = copy.copy(current_pred)[sort_idx]

                sk = 5
                sp = int(round(10*np.sqrt(float(size)/self.training_size)))

                s = UnivariateSpline(x, y, s=sp, k=sk, ext=3)
                y = s(x)
                y = np.maximum(y, 0.0)  
                current_pred[sort_idx] = y

            if self.renormalize:
                current_pred = current_pred / self.estimate_volumes[i]

            print(np.shape(current_pred))
            net_preds.append(current_pred)

        #if self.version == 1:
        #    tf.compat.v1.reset_default_graph()

        return net_preds


