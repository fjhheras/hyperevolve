# hyperevolve
Extremely simple evolutionary optimization of functions. 

It is intended to use in functions that take a long time to run, so running time is not a constrain. I use it to find the best hyperparameters in ML programs, hence the name hyperevolve.

Usage
-----

.. code:: python

    # We want to optimise this function
    def f_to_optimise(args):
        time.sleep(.1)  # Difficult to compute
        value = - (args['x'])**2 - 0.1*args['y']**2 + args['z']
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
