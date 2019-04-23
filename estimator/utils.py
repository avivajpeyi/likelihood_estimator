#!/usr/bin/env python3
"""
Util functions
"""

__author__ = 'Avi'
__version__ = '0.1.0'

import functools


def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


# function to run shell comands
def execute_in_shell(command=None, verbose=False):
    """
        command -- keyword argument, takes a list as input
        verbsoe -- keyword argument, takes a boolean value as input

        This is a function that executes shell scripts from within python.

        Keyword argument 'command', should be a list of shell commands.
        Keyword argument 'versboe', should be a boolean value to set verbose level.

        Example usage: execute_in_shell(command = ['ls ./some/folder/',
                                                    ls ./some/folder/  -1 | wc -l'],
                                        verbose = True )

        This command returns dictionary with elements: Output and Error.

        Output records the console output,
        Error records the console error messages.

    """
    error = []
    output = []

    if isinstance(command, list):
        for i in range(len(command)):
            try:
                process = subprocess.Popen(
                    command[i], shell=True, stdout=subprocess.PIPE
                )
                process.wait()
                out, err = process.communicate()
                error.append(err)
                output.append(out)
                if verbose:
                    print('Success running shell command: {}'.format(command[i]))
            except Exception as e:
                print('Failed running shell command: {}'.format(command[i]))
                if verbose:
                    print(type(e))
                    print(e.args)
                    print(e)

    else:
        print('The argument command takes a list input ...')
    return {'Output': output, 'Error': error}


def benchmark(func, args, n_run):
    """
    Parameters
    ----------
    func: function, the function to be benchmarked
    args: tuple, the arguments to be passed to the function
    n_run: int, the number of times the function should be run

    Returns
    -------
    List[int] of times it takes for the function to run
    """
    times = []
    for _ in range(n_run):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        func(*args)
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))  # milliseconds
    return times
