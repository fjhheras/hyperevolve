import copy
import itertools
import random

import numpy as np
from tqdm import tqdm


def truncated_normal(mu, std, tmax=-np.inf, tmin=np.inf):
    assert tmax > tmin
    while True:
        x = random.normalvariate(mu, std)
        if tmin < x < tmax:
            return x


class Element:
    def __init__(self, args, opt_args):
        self.args = copy.deepcopy(args)
        self.opt_args = opt_args

    def reset(self):
        new_args = {}
        for key in self.opt_args:
            new_value = random.uniform(self.opt_args[key]["min"],
                                       self.opt_args[key]["max"])
            if self.opt_args[key].get('int', False):
                new_value = int(round(new_value))
            new_args[key] = new_value

        self.args.update(new_args)

    def mutate(self, fraction):
        for key in self.opt_args:
            std = fraction*(self.opt_args[key]["max"] -
                            self.opt_args[key]["min"])
            oldvalue = self.args[key]
            new_value = truncated_normal(oldvalue, std,
                                         tmax=self.opt_args[key]["max"],
                                         tmin=self.opt_args[key]["min"])
            if self.opt_args[key].get('int', False):
                new_value = int(round(new_value))
            self.args[key] = new_value

    def die_and_resurrect_as(self, other):
        self.args = copy.deepcopy(other.args)


class Population:
    def __init__(self, n, args, opt_args):
        self.args = args
        self.opt_args = opt_args
        self.n = n
        self.elements = [Element(args, opt_args) for _ in range(n)]
        self.reset()

    def state(self):
        state = []
        for element in self.elements:
            state.append({key: element.args[key] for key in self.opt_args})
        return state

    def reset(self):
        for e in self.elements:
            e.reset()

    def step(self, results, std_fraction, n):
        self._select(results, n)
        self._mutate(std_fraction)

    def _select(self, results, n):
        sorted_index = sorted(range(len(results)), key=lambda k: results[k])
        for i, j in itertools.islice(zip(sorted_index, reversed(sorted_index)),
                                     n):
            self.elements[i].die_and_resurrect_as(self.elements[j])

    def _mutate(self, std_fraction):
        for element in self.elements:
            element.mutate(std_fraction)


class Annealer:
    def __init__(self, first, last, factor):
        self.first, self.last, self.factor = first, last, factor
        self.current = self.first

    def step(self):
        self.current = self.last + (self.current - self.last)*self.factor

    def reset(self):
        self.current = self.first


class EvolutionaryOptimiser:
    def __init__(self, goal, population,
                 std_annealer=None, n_to_select=1):
        self.goal = self._decorate_goal_fuction(goal)
        self.n_to_select = n_to_select
        self.population = population
        if std_annealer is None:
            self.std_annealer = Annealer(0.05, 0.01, 0.9)

    @staticmethod
    def _decorate_goal_fuction(goal_function):
        def new_goal_function(element):
            return goal_function(element.args)
        return new_goal_function

    def step(self):
        results = []
        for element in tqdm(self.population.elements):
            results.append(self.goal(element))
        self.population.step(results, self.std, self.n_to_select)
        self.std_annealer.step()
        return results, self.population.state()

    def start(self, iterations=100):
        for e in range(iterations):
            results, state = self.step()
            results = np.array(results)
            print("Mean value ", results.mean())
            best = results.argmax()
            best_state = state[best]
            print("Best ", best_state, " with value ", results[best])

    @property
    def std(self):
        return self.std_annealer.current
