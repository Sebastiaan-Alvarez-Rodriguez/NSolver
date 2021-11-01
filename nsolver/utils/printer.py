# The greater purpose of (functions in) this file is
# to convert strings to colored strings, which helps
# navigating the commandline interface


from enum import Enum
import os
import builtins

# Overridden print function to always flush.
# We need this practically everywhere when using ssh or multiprocessing.
def print(*args, **kwargs):
    kwargs['flush'] = True
    return builtins.print(*args, **kwargs)


class Color(Enum):
    '''An enum to specify what color you want your text to be'''
    RED = '\033[1;31m'
    GRN = '\033[1;32m'
    YEL = '\033[1;33m'
    BLU = '\033[1;34m'
    PRP = '\033[1;35m'
    CAN = '\033[1;36m'
    CLR = '\033[0m'


def printc(*args, **kwargs):
    '''Print given text with given color, if supported.'''
    if os.name == 'posix':
        s = ''.join(arg.value if isinstance(arg, Color) else arg for arg in args)
        if args[-1] != Color.CLR:
            s += Color.CLR.value
    else:
        s = ''.join(arg for arg in args if not isinstance(arg, Color))
    print(s, **kwargs)

def printn(string, color=Color.CAN, **kwargs):
    '''Print given note text'''
    printc(color, '[NOTE]', Color.CLR, ' ', string, **kwargs)

def prints(string, color=Color.GRN, **kwargs):
    '''Print given success text.'''
    printc(color, '[SUCCESS]', Color.CLR, ' ', string, **kwargs)

def printw(string, color=Color.YEL, **kwargs):
    '''Print given warning text'''
    printc(color, '[WARNING]', Color.CLR, ' ', string, **kwargs)

def printe(string, color=Color.RED, **kwargs):
    '''Print given error text'''
    printc(color, '[ERROR]', Color.CLR, ' ', string, **kwargs)