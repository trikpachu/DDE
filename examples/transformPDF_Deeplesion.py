'''
Generates and samples the PDF from the Imagenet dataset. Data is expected to be single directories per volume, containing imagefiles.  
The probability values at arbitrary points are interpolated from the surrounding values by linear interpolation.
Dumps a list per sample in a directory, containing [samples] or [samples, grid_samples], where each is of size [n_dim, n_samples]

Args:
    data_dir: STRING, Directory with the images. data_dir is directly searched with glob.glob, thus '/*' is appended to the directory, if not already there. defaults to  data/DeepLesion/*
    size: INT, size of the drawn sample distribution
    with_grid: BOOL, if True a grid with the corresponding true probability values is also generated (for nicer plots)
    grid_number: INT, number of samples in the grid
'''

import argparse
from dde.PDF_Generation import Prob_dist_from_3D as pdf3

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=1000, help='INT, size of the drawn sample distribution')
parser.add_argument('--data_dir', default='data/DeepLesion/*', help='STRING, Directory with the images. data_dir is directly searched with glob.glob, thus /* is appended to the directory, if not already there. defaults to  data/Stock_Data/*/*')
parser.add_argument('--grid_number', type=int, default=100, help='INT, number of samples in the grid')
parser.add_argument('--verbose', action='store_true', help='adds some verbose information')
parser.add_argument('--with_grid', action='store_true', help='BOOL, if True a grid with the corresponding true probability values is also generated (for nicer plots)')
args = parser.parse_args()

size = args.size
data_dir = args.data_dir
grid_number = args.grid_number
verbose = args.verbose
with_grid = args.with_grid



def main():
    pdf3(size=size, data=None, with_grid=with_grid, grid_number=grid_number, verbose=verbose, data_dir=data_dir, readimg=True, savedir='data/deeplesion_pdf').get_pdf()

if __name__ == "__main__":
    main() 