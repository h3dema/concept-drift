#!/usr/bin/python
"""
    runs the agent:
    python3 agent.py


    the --double-trick parameter uses the trick suggested by xxx, since MAB was not meant to run forever.
    If it is active, time periods of T iterations will be considered,
    and for each T iteractions this period is increased to 2T.
    --T define the initial period.

"""
__author__ = "Henrique Moura"
__copyright__ = "Copyright 2018, h3dema"
__credits__ = ["Henrique Moura"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Henrique Moura"
__email__ = "h3dema@gmail.com"
__status__ = "Production"
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import pickle

from mab import UCBAbstract


#
# set log
#
LOG = logging.getLogger('AGENT')


def softmax(x):
    _softmax = np.exp(x) / sum(np.exp(x))
    return _softmax


def code_action(c, p, num_channels=11):
    return int(p * (num_channels - 1) + c)


def decode_action(v, num_channels=11):
    c = v % num_channels
    p = v // num_channels
    return c, p


#
# this is the real class
# you should implement only the run_action method
# this method interacts with the environment, performing the action and collection the reward
# it returns if the agent was able to perform the action
#
class MABAgent(UCBAbstract):

    def __init__(self, n_actions):
        super().__init__(n_actions)

    def run_action(self, action):
        """
            :return r: the reward of the action taken
            :return success: boolean value indicating if the agent could perform the action or not
        """
        # nothing to run in simulation
        pass

    def prob_action(self, action):
        """return the probability of selecting the action"""
        p = softmax(self.w)
        return p[action]


def get_concept_drift(data, n_train=150, window=16, main_dir='..'):
    if main_dir not in sys.path:
        sys.path.append(main_dir)
    from calculate import calculate_drift

    y = data['r']
    y = np.sign(np.concatenate(([1], y[1:].values - y[:-1].values)))
    y[y == -1] = 0
    X = data[['Active time', 'Medium busy', 'channel',
              'new Active time', 'new Busy time', 'new Medium busy',
              'new_channel', 'new_txpower', 'txpower']].values
    result = calculate_drift(X, y, n_train=n_train, w=window, clfs_label=["AdWin"])
    detected_points = result['clfs']["AdWin"].get('detected_points', [])
    if len(detected_points) > 0:
        detected_points = [x for x, _ in detected_points]  # we only need the x (iteration number)
    LOG.info("Detected {} drift points: {}".format(len(detected_points), detected_points))
    return detected_points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the RL agent')
    # arg "n-actions" considers 15 power setups
    parser.add_argument('--n-actions', type=int, default=165, help='Inform the number of actions the RL agent can perform')
    parser.add_argument('--double-trick', type=bool, default=True, help='Perform the double trick in the timestep')
    parser.add_argument('--T', type=int, default=2, help='initial value for double trick')
    parser.add_argument('--debug', action="store_true", help='log debug info')

    parser.add_argument('--data', type=str, default='../sarss.h5', help='data to simulate')
    parser.add_argument('--iteractions', type=str, default='iteractions.p', help='results from simulation')

    args = parser.parse_args()

    if args.n_actions is None:
        LOG.info("You should define the number of actions to execute")
        parser.print_help()
        sys.exit(1)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    agent = MABAgent(n_actions=args.n_actions)

    data_all = pd.read_hdf(args.data, key='sarss')
    # process only traffic from both stations going to google
    data = data_all[data_all['sites'] == ('google', 'google')]
    columns_to_convert = ['new_channel', 'new_txpower', 'r']
    # generates warning >> SettingWithCopyWarning
    data.loc[:, columns_to_convert] = data.loc[:, columns_to_convert].astype('float')

    drifts = get_concept_drift(data)  # get drift points from data

    t = 1
    T = args.T
    __iterations = []
    for __iter in range(data.shape[0]):
        __d = data.iloc[__iter, :]
        """
            a) get data from real execution, including the action performed --ea
            b) get action proposed by the algorithm -- a
            c) compare result: ea == a ?
            d) probability of selecting ea?
            e) concept drift? yes, mark this point
            f) update system using pa, so the system learns the real environment

        """
        new_channel = float(__d['new_channel'])
        new_txpower = float(__d['new_txpower'])

        a = agent.get_action()  # best action using the current knowledge
        ea = code_action(new_channel, new_txpower)  # action performed

        Pa = agent.prob_action(a) * 100
        Pea = agent.prob_action(ea) * 100

        r = __d['r']  # reward received
        drift = __iter in drifts
        LOG.info('t: {} ch{} pwr {} Estimated action: {}[P={}] Actual action: {}[P={}] Reward: {} Drift {}'.format(t, new_channel, new_txpower, ea, Pea, a, Pa, r, drift))

        __iterations.append([__iter, t, new_channel, new_txpower, ea, Pea, a, Pa, r, drift])
        # don't need to run_action
        # r, success = agent.run_action(a)
        agent.update(ea, r)  # update using the executed action in order to learn

        t += 1
        if args.double_trick and t > T:
            t = 1
            agent.reset_pulls()
            try:
                T = 2 * T
            except OverflowError:
                T = args.T

    # save data
    pickle.dump(__iterations, open(args.iteractions, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
