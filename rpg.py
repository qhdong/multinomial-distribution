# -*- coding: utf-8 -*-
import numpy as np
from multinomial import MultinomialDistribution


class MagicItemDistribution(object):
    """rpg游戏中杀死怪物后掉落魔法装备的概率采样"""

    stats_names = ('dexterity', 'constitution', 'strength',
                   'intelligence', 'wisdom', 'charisma')

    def __init__(self, bonus_probs, stats_probs, rso=np.random):
        """初始化魔法装备的随机分布

        :param bonus_probs: 奖励的概率
        :param stats_probs: 奖励在玩家的各个属性中的分布
        """
        self.bonus_dist = MultinomialDistribution(bonus_probs, rso=rso)
        self.stats_dist = MultinomialDistribution(stats_probs, rso=rso)

    def _sample_bonus(self):
        """采样奖励"""

        # 只有一个事件发生
        sample = self.bonus_dist.sample(1)

        bonus = np.argmax(sample)
        return bonus

    def _sample_stats(self):
        """采样所有的奖励，以及奖励在不同的属性中如何分布"""

        bonus = self._sample_bonus()
        stats = self.stats_dist.sample(bonus)
        return stats

    def sample(self):
        """采样获得一个随机的魔法装备"""

        stats = self._sample_stats()
        item_stats = dict(zip(self.stats_names, stats))
        return item_stats

    def log_pmf(self, item):
        """计算给定的魔法装备的概率质量函数对数"""

        stats = np.array([item[stat] for stat in self.stats_names])
        log_pmf = self._stats_log_pmf(stats)
        return log_pmf

    def pmf(self, item):
        """计算给定装备的概率质量函数"""

        return np.exp(self.log_pmf(item))

    def _stats_log_pmf(self, stats):

        total_bonus = np.sum(stats)
        logp_bonus = self._bonus_log_pmf(total_bonus)
        logp_stats = self.stats_dist.log_pmf(stats)

        log_pmf = logp_bonus + logp_stats
        return log_pmf

    def _bonus_log_pmf(self, bonus):

        if bonus < 0 or bonus >= len(self.bonus_dist.p):
            return -np.inf

        x = np.zeros(len(self.bonus_dist.p))
        x[bonus] = 1

        return self.bonus_dist.log_pmf(x)