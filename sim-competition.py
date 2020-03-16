import numpy as np
from matplotlib import pyplot as plt


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
    data = np.asarray([sim_trial(n_players, money) for i in range(n_trials)]).T
    player_money = np.zeros(n_players)
    lucky = np.argmax(data, axis=0)
    u, c = np.unique(lucky, return_counts=True)
    player_money[u] -= money * c
    player_money += np.sum(data, axis=1)
    return player_money


if __name__ == '__main__':
    data = [sim_player_money(np) for np in range(3, 22, 3)]
    n = len(data)
    for i in range(n):
        fig, ax = plt.subplots()
        np = data[i].size
        ax.bar(list(range(np)), data[i])
        ax.set_xlabel('player order')
        ax.set_ylabel('money left')
        ax.set_title('Red bag competition for {} players'.format(np))
        plt.savefig('competition-{}-players.png'.format(np))
