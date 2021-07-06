'''
Generates and samples the PDF from the Imagenet dataset. Data is expected to be imagefiles in 8bit png format.  
The probability values at arbitrary points are interpolated from the surrounding values by linear interpolation.
Dumps a list per sample in a directory, containing [samples] or [samples, grid_samples], where each is of size [n_dim, n_samples]

Args:
    data_dir: STRING, Directory with the images. data_dir is directly searched with glob.glob, thus '/*' is appended to the directory, if not already there. defaults to  data/Imagenet/*
    size: INT, size of the drawn sample distribution
    with_grid: BOOL, if True a grid with the corresponding true probability values is also generated (for nicer plots)
    grid_number: INT, number of samples in the grid
'''

import argparse
from deep_density_estimation.PDF_Generation import Prob_dist_from_2D as pdf2

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=1000, help='INT, size of the drawn sample distribution')
parser.add_argument('--data_dir', default='data/Imagenet/*', help='STRING, Directory with the images. data_dir is directly searched with glob.glob, thus /* is appended to the directory, if not already there. defaults to  data/Stock_Data/*/*')
parser.add_argument('--grid_number', type=int, default=200, help='INT, number of samples in the grid')
parser.add_argument('--verbose', action='store_true', help='adds some verbose information')
parser.add_argument('--with_grid', action='store_true', help='BOOL, if True a grid with the corresponding true probability values is also generated (for nicer plots)')
args = parser.parse_args()

size = args.size
data_dir = args.data_dir
grid_number = args.grid_number
verbose = args.verbose
with_grid = args.with_grid



def main():
    pdf2(size=size, data=None, with_grid=with_grid, grid_number=grid_number, verbose=verbose, data_dir=data_dir, readimg=True, savedir='data/imagenet_pdf', imagenorm=255).get_pdf()

if __name__ == "__main__":
    main() 