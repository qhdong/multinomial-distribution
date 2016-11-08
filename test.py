# -*- coding: utf-8 -*-
import numpy as np
from rpg import MagicItemDistribution


if __name__ == '__main__':
    bonus_probs = np.array([0.0, 0.55, 0.25, 0.12, 0.06, 0.02])
    stats_probs = np.ones(6) / 6.0
    rso = np.random.RandomState(234892)
    item_dist = MagicItemDistribution(bonus_probs, stats_probs, rso)

    print(item_dist.sample())
    print(item_dist.sample())
    print(item_dist.sample())
    item = item_dist.sample()
    print(item_dist.pmf(item))