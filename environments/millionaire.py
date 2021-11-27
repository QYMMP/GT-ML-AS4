import sys

import numpy as np
from gym import utils
from gym.envs.toy_text import discrete
from six import StringIO

QUIT = 0
PLAY = 1

probability = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
reward = [0, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 1.5, 2.5, 5, 10]


class MillionaireEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.shape = (1, 11)
        self.start_state_index = np.ravel_multi_index((0, 0), self.shape)
        self.end_state_index = np.ravel_multi_index((0, 10), self.shape)

        self.desc = np.asarray([
            "SRRRRRRRRRG",
        ], dtype='c')

        nS = 11
        nA = 2

        P = {}
        for s in range(nS):
            # position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(nA)}
            P[s][QUIT] = self._calculate_transition_prob(s, QUIT)
            P[s][PLAY] = self._calculate_transition_prob(s, PLAY)

        isd = np.zeros(nS)
        isd[self.start_state_index] = 1.0

        super(MillionaireEnv, self).__init__(nS, nA, P, isd)

    def _calculate_transition_prob(self, current, delta):
        """
        Determine the outcome for an action. .
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position for transition
        :return: (1.0, new_state, reward, done)
        """
        if current == 10:
            return [(1.0, self.end_state_index, reward[current], True)]
        if delta == QUIT:
            return [(1.0, self.end_state_index, reward[current], True)]
        elif delta == PLAY:
            return [(probability[current], current + 1, 0, False),
                    (1.0 - probability[current], self.end_state_index, 0, True)]

    def render(self, mode="human"):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // 11, self.s % 1
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(
                ["Quit", "Play"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            return outfile

    def colors(self):
        return {
            b'S': 'green',
            b'R': 'lightslategray',
            b'G': 'gold'
        }

    def directions(self):
        return {
            1: 'P',
            0: 'Q'
        }

    def new_instance(self):
        return MillionaireEnv()
