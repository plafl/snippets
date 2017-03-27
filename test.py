import warnings
import itertools
from abc import ABCMeta, abstractmethod

import numpy as np


def add_constraints(constraints):
    """Given a list of constraints combine them in a single one.

    A constraint is a function that accepts a selection and returns True
    if the selection is valid and False if not.
    """
    if constraints is None:
        return lambda s: True
    else:
        return lambda s: all(constraint(s) for constraint in constraints)


class BasePolicyFamily(metaclass=ABCMeta):
    @abstractmethod
    def log_prob(self, s, params):
        """Compute the log probability of the selection 's' given the
        parameters 'params'"""
        pass

    @abstractmethod
    def jac_log_prob(self, s, params):
        """Compute the jacobian of the log probability relative to the
        parameters evaluated at the selection 's' and the parameters
        'params'"""
        pass

    @abstractmethod
    def sample(self, params):
        """Extract just one random selection.

        The result should be a numpy array with as many elements as replicas.
        The i-th component represents the hard drive selected for the
        i-th replica.
        """
        pass

    @abstractmethod
    def params_point(self):
        """Give an example of valid parameters.

        This method is used as an starting point for optimization methods
        and testing.
        """
        pass

    def normalize_params(self, params):
        """Return new params making sure they fall betwenn valid bounds

        The new params should be as close as possible as the ones provided.
        For example it the parameters represent a probability distribution
        they should sum to 1.
        """
        return params

    def sample_constrained(self, params, size, constraints=None):
        samples = []
        g = add_constraints(constraints)
        while len(samples) < size:
            sample = self.sample(params)
            if g(sample):
                samples.append(sample)
        return np.stack(samples)

    def loss(self, params, constraints=None):
        g = add_constraints(constraints)
        z_g = 0
        z_u = np.zeros(len(self.c))
        jac_z_g = np.zeros_like(params)
        jac_z_u = np.zeros((len(self.c), len(params)))
        for s in itertools.permutations(np.arange(len(self.c)), self.R):
            if g(s):
                p = np.exp(self.log_prob(s, params))
                q = p*self.jac_log_prob(s, params)
                z_g += p
                jac_z_g += q
                for i in s:
                    z_u[i] += p
                    jac_z_u[i, :] += q
        L = (np.log(self.c) - np.log(z_u/z_g/self.R)) @ self.c
        J = -((jac_z_u.T/z_u).T - jac_z_g/z_g).T @ self.c
        return L, J

    def build_selector(self, params, constraints=None):
        """Builds a function accepting no arguments that returns a valid selection.

        A selection is represented by an array with as many components as hard
        drives.
        A zero entry means the hard drive is unused, otherwise it says what
        replica is stored there.
        """
        g = add_constraints(constraints)

        def selector():
            sample = None
            while sample is None:
                candidate = self.sample(params)
                if g(candidate):
                    sample = candidate
            return sample

        return selector


class SimplePolicyFamily(BasePolicyFamily):
    def __init__(self, capacities, n_replicas):
        self.c = np.array(capacities) / np.sum(capacities)
        self.R = n_replicas
        # Initialize the probability of choosing the first hard drive
        L = self.c < 1/(self.R*len(self.c))
        M = np.logical_not(L)
        self.p_1 = np.empty_like(self.c)
        self.p_1[L] = self.R*self.c[L]
        self.p_1[M] = 1/np.sum(M)*(1 - np.sum(self.p_1[L]))

    def params_point(self):
        return np.copy(self.p_1)

    def normalize_params(self, params):
        return params/np.sum(params)

    def log_prob(self, s, params):
        p_2 = params
        logP = np.log(self.p_1[s[0]])
        for r in range(1, self.R):
            logP += np.log(p_2[s[r]])
            d = 1
            for i in range(r):
                d -= p_2[s[i]]
            logP -= np.log(d)
        return logP

    def jac_log_prob(self, s, params):
        p_2 = params
        jac = np.zeros_like(self.c)
        for r in range(1, self.R):
            jac[s[r]] += 1.0/p_2[s[r]]
            d = 1
            jac_d = np.zeros_like(self.c)
            for i in range(r):
                d -= p_2[s[i]]
                jac_d[s[i]] -= 1
            jac -= jac_d/d
        return jac

    def sample(self, params):
        p_2 = params
        i = np.random.choice(len(self.p_1), p=self.p_1)
        selection = [i]
        p = np.copy(p_2)
        for r in range(1, self.R):
            p[i] = 0
            p /= np.sum(p)
            i = np.random.choice(len(p), p=p)
            selection.append(i)
        return np.array(selection)


def optimal_params(policy_family, start=None,
                   constraints=None, step=1e-2, eps=1e-3,
                   max_iter=10000, verbose=0):
    """Apply gradient descent to find the optimal policy"""
    if start is None:
        start = policy_family.params_point()

    def loss(params):
        return policy_family.loss(params, constraints)

    params_old = np.copy(start)
    loss_old, jac_old = loss(params_old)
    it = 0
    while True:
        params_new = policy_family.normalize_params(params_old - step*jac_old)
        loss_new, jac_new = loss(params_new)
        jac_norm = np.sqrt(np.sum(jac_old**2))
        if loss_new > loss_old or jac_norm < eps:
            # converged
            break
        else:
            loss_old, jac_old = loss_new, jac_new
            params_old = params_new
            if it > max_iter:
                warnings.warn('max iter')
                break
        it += 1
        if verbose:
            print('it={0:>5d} jac norm={1:.2e} loss={2:.2e}'.format(
                it, jac_norm, loss_old))
    if verbose:
        print('Converged to desired accuracy :)')
    return params_old
