import numpy as np
from scipy import stats
from wechat_red_bag_simulation.sim_competition import RedBag


def sim_trial(n_bags: int, money: float):
    rb = RedBag(n_bags, money)
    trials = [rb.get_money() for _ in range(n_bags)]
    return trials


if __name__ == '__main__':
    data_true = np.loadtxt('../trials.csv')
    n_trials = 200
    n_players = 10
    money = 66.0
    np.random.seed(1024)
    data = np.asarray([sim_trial(n_players, money) for i in range(n_trials)]).T
    ks_stats = [stats.ks_2samp(data_true[i][data_true[i] != 0],
                               data[i][data[i] != 0])
                for i in range(data_true.shape[0])]
    ks_stats = np.asarray(ks_stats)

    md_str = """
    |order| p |reject?|
    |-----|---|-------|
    |0    |{0:02}%|  {10}|
    |1    |{1:02}%|  {11}|
    |2    |{2:02}%|  {12}|
    |3    |{3:02}%|  {13}|
    |4    |{4:02}%|  {14}|
    |5    |{5:02}%|  {15}|
    |6    |{6:02}%|  {16}|
    |7    |{7:02}%|  {17}|
    |8    |{8:02}%|  {18}|
    |9    |{9:02}%|  {19}|
    """

    ps = (ks_stats[:, 1] * 100).astype(int)
    md_str = md_str.format(*ps, *(ps < 1))
    # Markdown(md_str)
    print(md_str)
