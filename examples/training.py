###
# This is a wrapper script for the train class of DDE
###

import argparse
import numpy as np
import pickle
from dde.train import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=512, help='batch size for the neural nets [default = 32]')
parser.add_argument('--num_point', type=int, default=128, help='number of drawn neighbors [default = 128]')
parser.add_argument('--model', default='model_4',
                    help='neural net model to use, located in Models.py [default = model_4]')
parser.add_argument('--max_epoch', type=int, default=1000, help='maximum number of epochs [default = 1000]')
parser.add_argument('--decay_step', type=int, default=200000, help='decaystep of learning rate [default = 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='decayrate of learning rate [default = 0.7]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='base learning rate [default = 0.001]')
parser.add_argument('--not_renormalize', action='store_false', help='Refactoring the data?')
parser.add_argument('--data', default=None, help='name of the training datafile.')
args = parser.parse_args()

renormalize = args.not_renormalize
data = args.data
learning_rate = args.learning_rate
decay_step = args.decay_step
decay_rate = args.decay_rate
batch_size = args.batch_size
model = args.model
num_point = args.num_point
max_epoch = args.max_epoch


def main():
    # import the data
    trainer = trainer(batch_size=batch_size, num_point=num_point, model=model, max_epoch=max_epoch, 
                decay_step=decay_step, decay_rate=decay_rate, learning_rate=learning_rate, renormalize=renormalize, 
                training_data=data, validation_data=None, verbose=True)
    trainer.run()

if __name__ == "__main__":
    main()
