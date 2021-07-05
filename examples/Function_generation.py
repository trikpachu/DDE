import dde.PDF_Generation as g
import time
import argparse
import numpy as np
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dim', nargs='+', type=int, default=5, help='dim')
parser.add_argument('--size', nargs='+', type=int, default=5000, help='size')
parser.add_argument('--sets', nargs='+', type=str, default='all_sub', help='set')
parser.add_argument('--deselect', nargs='+', type=str, default=None, help='set')
parser.add_argument('--num_funcs', type=int, default=50, help='number of functions to generate')
parser.add_argument('--scale_max', type=int, default=10, help='number of functions to generate')
parser.add_argument('--more_random', action='store_true', help='whether to generate more time intensive but more random functions')
parser.add_argument('--naming', type=str, default='', help='Name Appendix of the generated files')

args = parser.parse_args()

naming = args.naming
NUM_FUNCS = args.num_funcs
SCALE_MAX = args.scale_max
more_random = args.more_random

def make_list(x):
    try:
        len(x)
        return x
    except:
        return [x]

sizes = make_list(args.size)
dims = make_list(args.dim)
sets = make_list(args.sets)
deselects = make_list(args.deselect)


if deselects != [None]:
    for size in sizes:
        for dim in dims:
            for s in range(len(sets)):
                setname = sets[s]
                deselect = deselects[s] 
                start = time.time()
                if more_random:
                    g.function_generation_more_time(size=size, num_funcs=NUM_FUNCS, complexity=[3,5], scale=[1,SCALE_MAX], dim=dim, select_functions=[setname], deselect_functions=[deselect], naming=naming).function_generation()
                else:
                    g.function_generation(size=size, num_funcs=NUM_FUNCS, complexity=[3,5], scale=[1,SCALE_MAX], dim=dim, select_functions=[setname], deselect_functions=[deselect], naming=naming).function_generation()
                print(f'test set generation for dim={dim}, size={size} and set={setname} took {time.time()-start} seconds')

else:
    for size in sizes:
        for dim in dims:
            for setname in sets: 
                start = time.time()
                if more_random:
                    g.function_generation_more_time(size=size, num_funcs=NUM_FUNCS, complexity=[3,5], scale=[1,SCALE_MAX], dim=dim, select_functions=[setname], naming=naming).function_generation()
                else:
                    g.function_generation(size=size, num_funcs=NUM_FUNCS, complexity=[3,5], scale=[1,SCALE_MAX], dim=dim, select_functions=[setname], naming=naming).function_generation()
                print(f'test set generation for dim={dim}, size={size} and set={setname} took {time.time()-start} seconds')
