###
# regard k stocks as k_armed bandits.
# take an action every day, buy 10^6 dollar in one stock at the start of the day, sell it at the end of the day.
# use market data
# reward: 10^6 * (close_price - open_price) / open_price
# best action is difficult to define. maybe we can take a moving window to calculate the average return, and define
#   best action as the stock with the highest return rate.

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
matplotlib.use('Agg')


class Bandit:
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    # @sample_averages: if True, use sample averages to update estimations instead of constant step size
    # @UCB_param: if not None, use UCB algorithm to select action
    # @gradient: if True, use gradient based bandit algorithm
    # @gradient_baseline: if True, use average reward as baseline for gradient based bandit algorithm
    # @best_action: with the highest return rate within the whole time
    def __init__(self, k_arm=10, epsilon=0., initial=0., step_size=0.1, sample_averages=False, UCB_param=None,
                 gradient=False, gradient_baseline=False, true_reward=0., ):
        self.k = k_arm
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial

    def reset(self):
        # real reward for each action is unknown

        # estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial

        # # of chosen times for each action
        self.action_count = np.zeros(self.k)

        self.time = 0

    # get an action for this bandit
    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        if self.UCB_param is not None:
            UCB_estimation = self.q_estimation + \
                self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)
            return np.random.choice(np.where(UCB_estimation == q_best)[0])

        if self.gradient:
            exp_est = np.exp(self.q_estimation)
            self.action_prob = exp_est / np.sum(exp_est)
            return np.random.choice(self.indices, p=self.action_prob)

        q_best = np.max(self.q_estimation)
        return np.random.choice(np.where(self.q_estimation == q_best)[0])

    # take an action, update estimation for this action
    def step(self, action, return_rate):
        # generate the reward under N(real reward, 1)
        reward = 10**6 * return_rate
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time

        if self.sample_averages:
            # update estimation using sample averages
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        elif self.gradient:
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            if self.gradient_baseline:
                baseline = self.average_reward
            else:
                baseline = 0
            self.q_estimation += self.step_size * (reward - baseline) * (one_hot - self.action_prob)
        else:
            # update estimation with constant step size
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
        return reward


def simulate(runs, time, bandits, return_rates):
    rewards = np.zeros((len(bandits), runs, time))
    # best_action_counts = np.zeros(rewards.shape)
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action, return_rates.iloc[t, action])
                rewards[i, r, t] = reward
                # if action == bandit.best_action:
                #     best_action_counts[i, r, t] = 1
    # mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_rewards


def get_return_rates(start_date, end_date, stocks):
    # @stocks: a list of stock names
    start_date = start_date
    end_date = end_date
    stocks_bf = web.DataReader(stocks, 'yahoo', start_date, end_date)
    return_rates = (stocks_bf['Open'] - stocks_bf['Close']) / stocks_bf['Open']
    return return_rates


start_date = dt.date(2018, 3, 1)
end_date = dt.date(2022, 3, 1)
stocks = ['IBM', 'AAPL', 'MNDT', 'RIG', 'AMD', 'AAL', 'AUY', 'F', 'CCL', 'NUAN']
# return_rates = get_return_rates(start_date, end_date, stocks)
# return_rates.to_csv('return_rates.csv')
return_rates = pd.read_csv('return_rates.csv')
return_rates.drop('Date',axis=1, inplace=True)

print('ok')
def figure_2_6(runs=2000, time=1000, return_rates=pd.DataFrame()):
    labels = ['epsilon-greedy-avg', 'epsilon-greedy-find-epsilon', 'epsilon-greedy-find-stepsize',
              'UCB', 'optimistic initialization']
    # labels = ['UCB', 'optimistic initialization']
    # generators = [lambda epsilon: Bandit(epsilon=epsilon, sample_averages=True),
    #               lambda epsilon: Bandit(epsilon=epsilon)]
    # parameters = [np.arange(0.1, 0.9, 0.1, dtype=np.float),
    #               np.arange(0.1, 0.9, 0.1, dtype=np.float)]
    generators = [lambda epsilon: Bandit(epsilon=epsilon, sample_averages=True),
                  lambda epsilon: Bandit(epsilon=epsilon),
                  lambda alpha: Bandit(epsilon=0.4, step_size=alpha),
                  lambda coef: Bandit(epsilon=0, UCB_param=coef, sample_averages=True),
                  lambda initial: Bandit(epsilon=0, initial=initial, step_size=0.1)]
    parameters = [np.arange(0.1, 0.9, 0.1, dtype=np.float),
                  np.arange(0.1, 0.9, 0.1, dtype=np.float),
                  np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8], dtype=np.float),
                  np.array([2**-4, 2**-3, 0.25, 0.5, 1, 2, 4, 8],dtype=np.float),
                  np.arange(-20000, 30000, 10000, dtype=np.float)]

    bandits = []
    for generator, parameter in zip(generators, parameters):
        for param in parameter:
            bandits.append(generator(param))

    average_rewards = simulate(runs, time, bandits, return_rates)
    rewards = np.mean(average_rewards, axis=1)

    i = 0
    j=1
    plt.figure(figsize=(10, 50))

    for label, parameter in zip(labels, parameters):
        l = len(parameter)
        plt.subplot(5, 1, j)
        plt.plot(parameter, rewards[i:i+l], label=label)
        i += l
        j += 1
        plt.xlabel('Parameter($x$)')
        plt.ylabel('Average reward')
        plt.legend()
    plt.savefig('test.png')
    plt.close()


figure_2_6(return_rates=return_rates)
# simulate(runs=500,time=500, bandits=[Bandit(gradient=True, step_size=0.1, gradient_baseline=True)],return_rates=return_rates)






