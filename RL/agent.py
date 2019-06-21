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
from scipy.special import softmax

from RL.mab import UCBAbstract


#
# set log
#
LOG = logging.getLogger('AGENT')


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


def code_action(c, p, num_channels=13):
    return p * num_channels + c


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the RL agent')
    # arg "n-actions" considers 15 power setups
    parser.add_argument('--n-actions', type=int, default=15, help='Inform the number of actions the RL agent can perform')
    parser.add_argument('--double-trick', type=bool, default=True, help='Perform the double trick in the timestep')
    parser.add_argument('--T', type=int, default=2, help='initial value for double trick')
    parser.add_argument('--debug', action="store_true", help='log debug info')

    parser.add_argument('--data', type=str, default='../sarss.h5', help='data to simulate')

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

    t = 1
    T = args.T
    for __d in data:
        """
            a) get data from real execution, including the action performed --ea
            b) get action proposed by the algorithm -- a
            c) compare result: ea == a ?
            d) probability of selecting ea?
            e) concept drift? yes, mark this point
            f) update system using pa, so the system learns the real environment

        """
        a = agent.get_action()
        Pa = agent.prob_action(a) * 100

        ea = code_action(__d['new_channel'], __d['new_txpower'])  # code the action using channel and power
        Pea = agent.prob_action(ea) * 100

        r = __d['r']  # reward received

        drift = False

        LOG.info('t: {} Action: {}[{}] Selected Action: {}[{}] Reward: {} Drift {}'.format(t, ea, Pea, a, Pa, r, drift))

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
