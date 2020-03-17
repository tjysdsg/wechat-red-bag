import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

__all__ = ['RedBag']


class RedBag:
    def __init__(self, n_bags: int, money: float):
        self.n_remain = self.n_bags = n_bags
        self.money_remain = self.money = money

    def get_money(self):
        """
        Get money from this red bag
        :return: how much money of a new red bag
        """
        assert self.money_remain >= 0
        if self.n_remain == 1:
            money = self.money_remain
            self.money_remain = 0
            self.n_remain = 0
        else:
            min_ = 0.01
            max_ = 2 * self.money_remain / self.n_remain
            money = np.random.uniform(0, 1.0, 1)[0] * max_
            money = max(min_, money)
            self.n_remain -= 1
            self.money_remain -= money
        return money


def sim_trial(n_bags: int, money: float):
    rb = RedBag(n_bags, money)
    trials = [rb.get_money() for _ in range(n_bags)]
    return trials


def sim_player_money(n_players=10, n_trials=10000, money=66.0):
    np.random.seed(1024)
    data = np.asarray([sim_trial(n_players, money) for _ in range(n_trials)]).T
    ranks = get_ranks(data)
    player_money = np.zeros(n_players)
    lucky = np.argmax(data, axis=0)
    u, c = np.unique(lucky, return_counts=True)

    # get remaining player money
    player_money[u] -= money * c
    player_money += np.sum(data, axis=1)

    # get number of luckiest for each player
    n_lucky = np.empty(n_players)
    n_lucky[u] = c
    return player_money, ranks, n_lucky


def get_ranks(array: np.ndarray):
    idx = array.argsort(axis=0)
    ranks = np.empty_like(idx)
    for i in range(ranks.shape[1]):
        ranks[idx[:, i], i] = np.arange(array.shape[0])
    return ranks


def normalize(data: np.ndarray, low=-1.0, high=1.0) -> np.ndarray:
    return low + (high - low) * (data - np.min(data)) / np.ptp(data)


if __name__ == '__main__':
    data = [sim_player_money(np) for np in range(3, 22, 3)]
    n = len(data)
    for i in range(n):
        bar_width = 0.35
        data_, _, lucky = data[i]
        # normalize data and lucky
        data_ = normalize(data_)
        lucky = normalize(lucky)
        fig, ax = plt.subplots()
        n_players = data_.size
        labels = np.asarray(list(range(n_players)))
        ax.bar(labels - bar_width / 2, data_, label='Remaining money', width=bar_width, color='#a6cee3')
        ax.bar(labels + bar_width / 2, lucky, label='Number of luckiest', width=bar_width, color='#edd1cb')

        ax.set_xlabel('player order')
        ax.set_ylabel('normalized value')
        ax.set_title('Red bag competition for {} players'.format(n_players))
        ax.set_xticks(labels)
        ax.set_xticklabels([str(i) for i in labels])
        plt.legend()
        fig.tight_layout()
        plt.savefig('competition-{}-players.png'.format(n_players))

    # for i in range(n):
    #     data_, ranks, lucky = data[i]
    #     mean_ranks = np.mean(ranks, axis=1)
    #     fig, ax = plt.subplots()
    #     sns.barplot(mean_ranks, data_, ax=ax, palette="Blues_d")
    #     ax.set_xlabel('money rank')
    #     ax.set_ylabel('money left')
    #     ax.tick_params(axis='x', which='major', labelsize=10)
    #     plt.xticks(rotation=-90)
    #     n_players = data_.size
    #     ax.set_title('competition money w.r.t ranks; {} players'.format(n_players))
    #     plt.savefig('competition-money-wrt-ranks-{}-players.png'.format(n_players))
