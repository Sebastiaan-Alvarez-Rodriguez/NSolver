
def run(evaluations, filters, paths, verbose=False):
    print('Searching for optimizers...')
    optimizers = get_optimizers(paths)
    if verbose:
        for k,v in optimizers.items():
            print(f'    Found {len(v):03d} optimizers in path {k}')
    prints(f'    Found {sum(x for x in optimizers.values())} optimizers.')
    return True