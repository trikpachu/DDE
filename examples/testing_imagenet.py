'''
This is a wrapper script for the train class of DDE to predict the estimates of the transformed imagenet data.
This script takes only care for data transformed without grid data generation. 
For that you'll need to adapt the data loading in this script.
'''

import argparse
import numpy as np
import pickle
from dde.estimate import estimator
import utils
import glob
import copy
import time
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=512, help='batch size for the neural nets [default = 32]')
parser.add_argument('--num_point', type=int, default=128, help='number of drawn neighbors [default = 128]')
parser.add_argument('--model', default='model_4', help='neural net model to use, located in Models.py [default = model_4]')
parser.add_argument('--model_path', default=None, help='trained model. Must be set for custom trained models. [default = None]')
parser.add_argument('--training_size', type=int, default=None, help='size of the training samples. Must be provided for custom trained models if it is not 5000. Otherwise it will be inferred from dimensionality (1000 for dim==1, 5000 else.). [default = None]')
parser.add_argument('--not_renormalize', action='store_false', help='Refactoring the data?')
parser.add_argument('--dist', default=None, help='directory containing the transformed stock data.')
args = parser.parse_args()

renormalize = args.not_renormalize
dist = args.dist
model = args.model
model_path = args.model_path
batch_size = args.batch_size
num_point = args.num_point
training_size = args.training_size


def main():
    # import the data
    data = []
    ys = []
    file_list = glob.glob(os.path.join(dist, '*.p'))
    net_time = 0
    net_mse = []
    net_rmse = []
    net_kl = []

    for fname in file_list:
        f = open(fname, 'rb')
        data_ = pickle.load(f) 

        sample = data_[:2]
        y = data_[2]

        data.append(copy.copy(sample))
        ys.append(y)

    start = time.time() 
    estim = estimator(batch_size=batch_size, num_point=num_point, model=model, model_path=model_path, training_size=training_size,
                 renormalize=renormalize, dist=data, estimate_at=None, with_nn=False, smoothing=False, verbose=True)
    estimate = estim.run()
    end = time.time()
    net_time = end-start

    for i in range(len(data)):
        mse = utils.mean_squared_error(ys[i], estimate[i])
        net_mse.append(mse)
        rmse = utils.relative_mean_squared_error(ys[i], estimate[i])
        net_rmse.append(rmse)
        kl = utils.kl_divergence(estimate[i], ys[i])
        net_kl.append(kl)
    
    print('obtained the following results:')
    print('     TIME:   ', net_time)
    print('     MSE:   ', np.mean(net_mse))
    print('     MSRE:   ', np.mean(net_rmse))
    print('     KL:   ', np.mean(net_kl))

if __name__ == "__main__":
    main()
