import random
import time
from hyperevolve import Population, EvolutionaryOptimiser

if __name__ == '__main__':
    # We want to optimise this function
    def f_to_optimise(args):
        time.sleep(.1)  # Difficult to compute
        value = - (args["x"])**2 - 0.1*args["y"]**2 + args['z']
        return value + 1e-3*random.normalvariate(0, 1)
    # It has 3 arguments, but we will optimise
    # on x and y. z will stay constant
    args = {'x': 0.89, 'y': -2.1, 'z': 3}
    # We want some bounds on x and y, and we want x integer
    opt_args = {'x': {'max': 5, 'min': -5, 'int': True},
                'y': {'max': 5, 'min': -5}}

    population = Population(9, args, opt_args)
    optimiser = EvolutionaryOptimiser(f_to_optimise, population, n_to_select=3)
    optimiser.start()
