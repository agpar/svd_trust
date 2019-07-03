import numpy as np
import random
from svd_base import SVDBase
from svd_trust import SVDTrust

NUM_FACTORS = 5
NUM_AGENTS = 100
SPARSITY = 0.1
RATE_NOISE_MEAN = 0
RATE_NOISE_STD = 0.1
MAX_RATING = 1.0
MIN_RATING = 0.0
AGENT_SELF_RATING = MAX_RATING

class Agent:
    def __init__(self, _id, factors):
        self._id = _id
        self.factors = np.array(factors)
        self._max_dist = self._max_dist()

    def rate(self, other, noise=True):
        """Return a rating for an agent based on factor dist + noise"""
        if len(self.factors) != len(other.factors):
            raise Exception("Can not rate when factors are different")

        dist = np.linalg.norm(self.factors - other.factors)
        if noise:
            dist += np.random.normal(RATE_NOISE_MEAN, RATE_NOISE_STD)

        if dist > MAX_RATING:
            dist = MAX_RATING
        if dist < MIN_RATING:
            dist = MIN_RATING

        return (self._id, other._id, MAX_RATING - dist)

    def _max_dist(self):
        f_len = len(self.factors)
        return np.linalg.norm(np.ones(f_len))

    def __eq__(self, other):
        return self._id == other._id


def gen_factors(f_len):
    facts = []
    for i in range(f_len):
        facts.append(random.random())
    facts = np.array(facts)
    mag = np.linalg.norm(facts)
    return facts / mag


def run(sparsity=SPARSITY):
    agents = []
    ratings = []
    for i in range(NUM_AGENTS):
        agents.append(Agent(i, gen_factors(NUM_FACTORS)))

    possible_ratings = np.array([(a1, a2) for a1 in agents for a2 in agents if a1 != a2])
    num_ratings_to_do = int(len(possible_ratings) * sparsity)

    # Add the self ratings
    for a in agents:
        ratings.append((a._id, a._id, AGENT_SELF_RATING))

    # Add the sparsity controlled ratings
    choice_idxs = np.random.choice(len(possible_ratings), num_ratings_to_do, replace=False)
    ratings_to_do = possible_ratings[choice_idxs]
    for a1, a2 in ratings_to_do:
        ratings.append(a1.rate(a2))

    return agents, ratings

def calc_error(agents, predictor):
    err = 0
    for a1 in agents:
        for a2 in agents:
            if a1 != a2:
                _, _, rate_no_noise = a1.rate(a2, noise=False)
                err += (rate_no_noise - predictor.predict(a1._id, a2._id)) ** 2

    return err / ((len(agents)** 2) - len(agents))

def sparsity_test():
    sparsities = [0.001 * i for i in range(1, 101)]
    errs = []
    for sp in sparsities:
        agents, ratings = run(sparsity=sp)

        svdb = SVDBase().learn(ratings, 10, 0.01, 0.1, 0.1, 100)
        svdt = SVDTrust().learn(ratings, 10, 0.01, 0.1, 0.3, 100)

        svdb_err = calc_error(agents, svdb)
        svdt_err = calc_error(agents, svdt)
        errs.append({
            'svdb_err': svdb_err,
            'svdt_err': svdt_err
        })
        print(f"Sparsity: {sp:.4f}, svdb_err: {svdb_err:.4f}, svdt_err: {svdt_err:.4f}")
    return errs
