#!/usr/bin/python3

import pstats
from pstats import SortKey
from sys import argv
from os import system

number_of_points = int(argv[1]) if len(argv) > 1 else 100
grid_granularity = int(argv[2]) if len(argv) > 2 else 5

filename = f"perf_{number_of_points}_{grid_granularity}"
system(f"python3 -m cProfile -o {filename} ./voronoi.py {number_of_points} {grid_granularity}")
p = pstats.Stats(filename)
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(10)
