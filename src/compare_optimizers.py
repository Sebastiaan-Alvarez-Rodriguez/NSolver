#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Script for comparing optimizers for the course 'Natural Computing'
at LIACS, Fall 2018.
Assumes it is located in a structure as follows:

    PA/                                # This name does not matter
    ├── src/                           # All files in src/ are given
    │   ├── compare_optimizers.py
    │   ├── magic_square.py
    │   ├── ga_skeleton.py
    │   └── sa_skeleton.py
    │
    ├── Submissions/                   # Place your implementations here
    │   ├── lastname1_lastname2_ga.py
    │   ├── lastname1_lastname2_sa.py
    │   └── reference_random.py        # A reference random search implementation
    │
    └── Results/                       # This directory is automatically created
        └── ...                        # and stores all generated results

Optimizers in 'Submissions/' are automatically detected if they adhere
to the following naming convention:

 * lastname_sa() in lastname_sa.py
 * lastname_ga() in lastname_ga.py
or
 * lastname1_lastname2_sa() in lastname1_lastname2_sa.py
 * lastname1_lastname2_ga() in lastname1_lastname2_ga.py

Each optimizer will be tested on the following three problems with a budget
of 10,000 evaluations:
 - 12x12 Magic Square
 - 7x7x7 Semi-Perfect Magic Cube
 - 7x7x7 Perfect Magic Cube

This script will automatically plot (in both .png and .pdf)
 - a comparison of the median convergence for all optimizers per problem
 - a visualization of the best found solution per optimizer and problem


Usage:
$ python3 src/compare_optimizers.py
$ python3 src/compare_optimizers.py help | -h | --help
$ python3 src/compare_optimizers.py [num_runs]
$ python3 src/compare_optimizers.py [num_runs] [name_filter]

help, -h, --help
    Prints this help text

[num_runs]
    Number of runs to perform per optimizer, default 3

[name_filter]
    Only runs optimizers where [name_filter] is contained in the filename,
    disabled by default


Examples:

$ python3 src/compare_optimizers.py
    Runs the script using default parameters, comparing all algorithms

$ python3 src/compare_optimizers.py 15
    Runs all algorithms 15 times to determine median performance

$ python3 src/compare_optimizers.py 5 ga
    Runs all algorithms with 'ga' in the name (that is, all GA's) 5 times
    to determine median performance



Author: Sander van Rijn <s.j.van.rijn@liacs.leidenuniv.nl>
Based on a previous Matlab file by Hao Wang and Koen van der Blom
Last modified: 2018-11-05
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import namedtuple
from importlib import import_module
from mpl_toolkits import mplot3d
from time import time
from warnings import warn

module_path = [
    os.path.abspath(os.path.join('.')),
    os.path.abspath(os.path.join('..')),
]
for mp in module_path:
    if mp not in sys.path:
        sys.path.append(mp)

import magic_square


Problem = namedtuple('Problem', ['name', 'fitness_func', 'dim', 'budget'])
Optimizer = namedtuple('Optimizer', ['func', 'fullname', 'algorithm_name', 'lastname_1', 'lastname_2'])


def compare_optimizers(optimizers, problems, *, num_runs=1, results_dir=None, show_plot=False):
    """ Compare the given optimizers on the given prolems.
        Generates convergence and solution plots if results_dir or show_plots is given.
    """

    tab = '    '
    run_width = len(str(num_runs))
    opt_width = len(str(len(optimizers)))
    prob_progress = 'Test problem {i}/{num} ({name})'.format(i='{i}', num=len(problems), name='{name}')
    opt_progress = '{indent}[{j:{width}d}/{num}] {opt}'
    stat_progress = '\r{indent}[{k}/{num}]'.format(indent=tab*2, k='{k:{width}d}', num=num_runs)
    stat_string = 'median fopt={fopt:10,.2F}, average runtime={elapsed:6.2F}s'

    for problem_idx, problem in enumerate(problems):
        print(prob_progress.format(i=problem_idx+1, name=problem.name))

        eval_budget = problem.budget
        if problem.dim == 2:
            n = 12
        else:
            n = 7

        xopt_tracking = np.zeros((len(optimizers), num_runs, n ** problem.dim))
        elapsed_tracking = np.zeros((len(optimizers), num_runs))
        hist_best_f_tracking = np.zeros((len(optimizers), num_runs, eval_budget))

        for optimizer_idx, optimizer in enumerate(optimizers):
            func = optimizer.func
            stats = ''
            print(opt_progress.format(indent=tab, j=optimizer_idx+1, width=opt_width,
                                      num=len(optimizers), opt=optimizer.fullname))
            for run_idx in range(num_runs):
                print(stat_progress.format(k=run_idx+1, width=run_width), stats, end='')
                sys.stdout.flush()

                start_time = time()
                xopt, fopt, hist_best_f = func(problem.dim, eval_budget, problem.fitness_func,
                                               return_stats=True, do_plot=False)

                xopt_tracking[optimizer_idx, run_idx, :] = xopt
                elapsed_tracking[optimizer_idx, run_idx] = time() - start_time
                hist_best_f_tracking[optimizer_idx, run_idx, :] = hist_best_f

                stats = stat_string.format(
                    fopt=np.median(hist_best_f_tracking[optimizer_idx, :run_idx+1, -1]),
                    elapsed=np.mean(elapsed_tracking[optimizer_idx, :run_idx+1])
                )

            print(stat_progress.format(k=num_runs, width=run_width), stats)

        print()
        if results_dir:
            np.savez(
                results_dir + problem.name,
                optimizers=[optimizer.fullname for optimizer in optimizers],
                hist_best_f_tracking=hist_best_f_tracking,
                xopt_tracking=xopt_tracking,
                elapsed_tracking=elapsed_tracking
            )

        plot_convergences(hist_best_f_tracking, optimizers, problem.name,
                          results_dir=results_dir, show_plot=show_plot)
        display_best_results(hist_best_f_tracking, xopt_tracking, optimizers, problem,
                             results_dir=results_dir, show_plot=show_plot)


def nice_name(optimizer):
    """ Returns a nicely formatted combination of the lastnames associated with an optimizer """
    opt = optimizer.algorithm_name
    if not optimizer.lastname_2:
        return '{opt} by {lname}'.format(opt=opt, lname=optimizer.lastname_1)
    return '{opt} by {lname1} and {lname2}'.format(opt=opt,
                                                   lname1=optimizer.lastname_1,
                                                   lname2=optimizer.lastname_2)


def plot_convergences(hist_best_f_tracking, optimizers, name, *, results_dir=None, show_plot=False):
    """ Plot the convergence history for all given optimizers """
    if not results_dir and not show_plot:
        return

    # Extend the default Matplotlib property cycle from just colors to include 4 linestyles
    color_cycler = plt.rcParamsDefault['axes.prop_cycle']
    style_cycler = plt.cycler('linestyle', ['-', '--', '-.', ':'])
    plt.rcParams['axes.prop_cycle'] = style_cycler * color_cycler

    author_names = [nice_name(opt) for opt in optimizers]

    plt.figure(figsize=(12,9))
    for median, authors in zip(np.median(hist_best_f_tracking, axis=1), author_names):
        plt.plot(median, label=authors)
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.ylabel('Error')
    plt.xlabel('Evaluations used')
    plt.title(name)
    plt.legend(bbox_to_anchor=(1.04,0.5), loc='center left', borderaxespad=0, ncol=2)
    plt.subplots_adjust(right=0.7)

    if results_dir:
        plt.savefig('{dir}{name}.png'.format(dir=results_dir, name=name), bbox_inches="tight")
        plt.savefig('{dir}{name}.pdf'.format(dir=results_dir, name=name), bbox_inches="tight")
    if show_plot:
        plt.show()


def display_best_results(hist_best_f_tracking, xopt_tracking, optimizers, problem, results_dir=None, show_plot=False):
    """ Display the best solution candidate found per optimizer """
    if not results_dir and not show_plot:
        return
    if problem.dim not in [2, 3]:
        raise ValueError("Invalid problem dimension, expecting problem.dim=2 or problem.dim=3")

    for optimizer_idx, optimizer in enumerate(optimizers):
        best_index = np.argmin(hist_best_f_tracking[optimizer_idx, :, -1])
        if problem.dim == 2:
            plot_square(xopt_tracking[optimizer_idx, best_index, :], optimizer,
                        results_dir=results_dir, show_plot=show_plot)
        elif problem.dim == 3:
            is_semi_perfect = 'Semi' in problem.name
            plot_cube(xopt_tracking[optimizer_idx, best_index, :], optimizer,
                      results_dir=results_dir, show_plot=show_plot, semi_perfect=is_semi_perfect)


def plot_square(square, optimizer, results_dir=None, show_plot=False):
    """ Plot the best magic square found per optimizer """
    if not results_dir and not show_plot:
        return

    n = int(np.sqrt(len(square)))
    square = square.reshape((n, n))
    X, Y = np.meshgrid(np.arange(n), np.arange(n))

    plt.figure(figsize=(9,7))
    for x, y in zip(X.flatten(), Y.flatten()):
        plt.text(x+.5, n-y-.5, s=str(int(square[y,x])),
                 color=plt.cm.winter(square[y,x]/n**2),
                 horizontalalignment='center',
                 verticalalignment='center')
    for x in np.arange(n):
        plt.axhline(x+1, color='black', linewidth=1)
        plt.axvline(x+1, color='black', linewidth=1)

    plt.title("Best square found using {name}".format(name=nice_name(optimizer)))
    plt.xticks([])
    plt.yticks([])
    plt.ylim(top=n, bottom=0)
    plt.xlim(left=0, right=n)
    plt.tight_layout()

    if results_dir:
        plt.savefig('{dir}Best_square_{name}.png'.format(dir=results_dir, name=optimizer.fullname))
        plt.savefig('{dir}Best_square_{name}.pdf'.format(dir=results_dir, name=optimizer.fullname))
    if show_plot:
        plt.show()


def plot_cube(cube, optimizer, results_dir=None, show_plot=False, *, semi_perfect=False):
    """ Plot the best (semi-perfect) magic cube found per optimizer """
    if not results_dir and not show_plot:
        return

    n = int(np.round(len(cube) ** (1 / 3)))
    cube = cube.reshape((n, n, n))
    X, Y, Z = np.meshgrid(np.arange(n), np.arange(n), np.arange(n))
    semi = 'semi-perfect ' if semi_perfect else ''

    fig = plt.figure(figsize=(9,7))
    ax = fig.gca(projection='3d')

    plt.title("Best {semi}cube found using {name}".format(semi=semi, name=nice_name(optimizer)), y=.91)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.zaxis.set_ticks([])
    ax.set_ylim(bottom=0, top=n)
    ax.set_xlim(left=0, right=n)
    ax.set_zlim(bottom=0, top=n)
    ax.view_init(elev=10, azim=-85)

    plt.tight_layout(pad=5, h_pad=0, w_pad=0, rect=[-.25,-.25,1.25,1.25])

    for x, y, z in zip(X.flatten(), Y.flatten(), Z.flatten()):
        ax.text(x+.5, y+.5, z+.5, s=str(int(cube[y,x,z])),
                color=plt.cm.winter(y/n), fontsize=7,
                horizontalalignment='center',
                verticalalignment='center')

    semi = 'semi-perfect_' if semi_perfect else ''
    if results_dir:
        plt.savefig('{dir}Best_{semi}cube_{name}.png'.format(dir=results_dir, semi=semi, name=optimizer.fullname))
        plt.savefig('{dir}Best_{semi}cube_{name}.pdf'.format(dir=results_dir, semi=semi, name=optimizer.fullname))
    if show_plot:
        plt.show()


def import_optimizer(name, path):
    """ Import the optimizer contained in the given filename and path """
    func = getattr(import_module('{path}.{name}'.format(path=path, name=name), path), name)
    parts = name.split('_')
    if len(parts) not in (2,3):
        raise ImportError('Name does not seem to match naming scheme')
    lastname1 = parts[0]
    lastname2 = parts[1] if len(parts) == 3 else None
    algorithm = parts[-1]
    return Optimizer(func, name, algorithm, lastname1, lastname2)


def gather_optimizers(path, *, name_filter=None):
    """ Collect all optimizers in the given path, optionally filtered by name_filter """
    source_files = (file[:-3] for file in os.listdir(path) if file.endswith('.py'))
    if name_filter:  # Filter for given algorithm only if given
        source_files = (file for file in source_files if name_filter in file)

    optimizers = []
    for name in source_files:
        try:
            optimizers.append(import_optimizer(name, path))
        except (AttributeError, ImportError):
            print(UserWarning(
                "Warning:\n"
                "  Optimizer '{name}' not found in \'{path}/{name}\'.\n"
                "  Please make sure you follow the specified naming convention.\n\n"
                "  Run 'python3 compare_optimizers.py --help' for more information\n"
                "".format(path=path, name=name)))

    return optimizers


def guarantee_folder_exists(path_name):
    """ Makes sure the given path exists after this call """
    try:
        os.mkdir(path_name)
    except OSError:
        pass  # Folder exists, nothing to be done


def run(num_runs=3, name_filter=None):

    # If our current working directory is src/, move up one so we can find Submissions/
    if os.path.basename(__file__) in os.listdir('.'):
        os.chdir('..')
    optimizers = gather_optimizers(path='Submissions', name_filter=name_filter)

    if len(optimizers) == 0:
        print('No optimizers detected!')
        exit()
    else:
        print('{num} optimizers detected\n'.format(num=len(optimizers)))

    eval_budget = 10000
    problems = [
        Problem('Magic Square',            magic_square.eval_square,    dim=2, budget=eval_budget),
        Problem('Semi-Perfect Magic Cube', magic_square.eval_semi_cube, dim=3, budget=eval_budget),
        Problem('Perfect Magic Cube',      magic_square.eval_cube,      dim=3, budget=eval_budget),
    ]

    show_plots = False
    results_dir = 'Results/'
    guarantee_folder_exists(results_dir)

    compare_optimizers(optimizers, problems, num_runs=num_runs,
                       show_plot=show_plots, results_dir=results_dir)



if __name__ == '__main__':

    if len(sys.argv) == 1:
        run()
    elif len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print(__doc__)
        exit(0)
    elif len(sys.argv) == 2:
        run(num_runs=int(sys.argv[1]))
    elif len(sys.argv) == 3:
        run(num_runs=int(sys.argv[1]), name_filter=sys.argv[2])
