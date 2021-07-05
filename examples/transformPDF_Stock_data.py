'''
Generates and samples the PDF from the Stock market dataset. Data is expected to be tabulated in txt/csv files with 1 sample point per row structured x,y.  
The probability values at arbitrary points are interpolated from the surrounding values by linear interpolation.
Dumps a list per sample in a directory (arg to get_pdf), containing [samples] or [samples, grid_samples], where each is of size [n_dim, n_samples]

Args:
    data_dir: STRING, Directory with the stock data files. data_dir is directly searched with glob.glob, thus '/*' is appended to the directory, if not already there. defaults to  data/Stock_Data/*/*
    size: INT, size of the drawn sample distribution
    with_grid: BOOL, if True a grid with the corresponding true probability values is also generated (for nicer plots)
    grid_number: INT, number of samples in the grid
'''

import numpy as np
import argparse
import os
import pandas as pd
from datetime import datetime as dt
import time
import glob
from dde.PDF_Generation import Prob_dist_from_1D as pdf1

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=1000, help='INT, size of the drawn sample distribution')
parser.add_argument('--data_dir', default='data/Stock_Data/*/*', help='STRING, Directory with the images. data_dir is directly searched with glob.glob, thus /* is appended to the directory, if not already there. defaults to  data/Stock_Data/*/*')
parser.add_argument('--grid_number', type=int, default=10000, help='INT, number of samples in the grid')
parser.add_argument('--verbose', action='store_true', help='adds some verbose information')
parser.add_argument('--with_grid', action='store_true', help='BOOL, if True a grid with the corresponding true probability values is also generated (for nicer plots)')
args = parser.parse_args()

size = args.size
data_dir = args.data_dir
grid_number = args.grid_number
verbose = args.verbose
with_grid = args.with_grid


def sinceEpoch(date): # returns seconds since epoch
    return time.mktime(date.timetuple())

def toYearFraction(el):
    '''
    Input: String with the format 'year-month-day'

    Returns the date as decimal
    '''
    date = dt.strptime(el, '%Y-%m-%d')
    
    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction

def get_data():
    if not data_dir.endswith('*'):
        os.path.join(data_dir, '*')
    file_list = glob.glob(data_dir)
    data = []
    for fname in file_list:
        try:
            raw_data = pd.read_csv(fname, sep=',', header=0).values
        except: continue
        x = []
        y = raw_data[:, 1]
        years = []
        
        for el in raw_data[:, 0]:
            val = toYearFraction(el)
            x.append(val)
        data.append([[i, j] for i, j in zip(x,y)])
    
    return data

def main():
    data = get_data()
    pdf1(size=size, data=data, with_grid=with_grid, grid_number=grid_number, verbose=verbose, readtxt=False, exclude_short=True, savedir='data/stocks_pdf/').get_pdf()

if __name__ == "__main__":
    main() 