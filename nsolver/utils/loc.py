import coco_grading.utils.fs as fs


def default_referencedir():
    '''Default path to coco-reference source directory.'''
    return fs.join(fs.dirname(fs.abspath()), 'coco-reference')

def default_extractdir():
    '''Default path to extract directory, where all discovered archives are extracted.'''
    return fs.join(fs.abspath(), 'extracted')

def default_resultsdir():
    '''Default path to results directory, where all raw gtest output is stored.'''
    return fs.join(fs.abspath(), 'results')

def default_gradingdir():
    '''Default path to grading directory, where all output reports are stored.'''
    return fs.join(fs.abspath(), 'grading')