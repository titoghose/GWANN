# coding: utf-8
import os
import datetime
import subprocess
import pandas as pd
import argparse


def vprint(*print_args):
    """Prints the passed arguments based on the verbosity level at which
    the program is run.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', type=int, default=0)
    parser.add_argument('--label', type=str)
    parser.add_argument('--base', type=str)
    parser.add_argument('--chrom', type=str)
    parser.add_argument('--sp', type=int, default=0)
    parser.add_argument('--ep', type=int, default=1000)
    parser.add_argument('--onlycovs', type=bool, default=False)
    parser.add_argument('--gene', type=str)
    parser.add_argument('--win', type=int)
    parser.add_argument('--win_size', type=int, default=50)
    parser.add_argument('--flanking', type=int, default=2500)
    
    args = parser.parse_args()

    if args.v:
        print(*print_args)
    else:
        return
