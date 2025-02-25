"""
This script aims to implement variable ankle limits in crocoddyl.
"""

import pinocchio as pin
import crocoddyl as croc
import numpy as np



# croc.ResidualModelAbstract
# croc.ResidualDataAbstract

"""
lower <  q_m = f(qs) + delta_m < range
qm = f(qs) + delta_m

dqmdqm = 0
dqmdqs = Ja
dqmdvm = 0
dqmdvs = 0
"""

class ResidualModelAnkleLimits(croc.ResidualModelAbstract):
    def __init__(self, state, nr, nu, ankleActuator1, ankleActuator2, delta_m):
        """
        Croccoddyl version of the residual model:
        r = m + delta_m = qm
        should be used with a quadratic barrier
        """
        super().__init__(state, nr, nu, True, True, True)

        self.ankleActuator1 = ankleActuator1
        self.ankleActuator2 = ankleActuator2
        self.nq = self.ankleActuator1.model.nq
        self.ankle1_idx_qs = self.ankleActuator1.idx_qs
        self.ankle2_idx_qs = self.ankleActuator2.idx_qs
        self.rx = np.zeros((nr,ankleActuator1.model.nq + ankleActuator1.model.nv))
        self.delta_m = delta_m
        assert len(delta_m) == nr

    def calc(self, data, x, u):
        self.ankleActuator1.actuation(x[:self.nq], np.zeros(self.ankleActuator1.nu))
        self.ankleActuator2.actuation(x[:self.ankleActuator2.model.nq], np.zeros(self.ankleActuator2.nu))
        self.m_1 = self.ankleActuator1.m
        self.m_2 = self.ankleActuator2.m
        data.r = np.array([self.m_1[0], self.m_2[0]]) + np.array(self.delta_m)

    def calcDiff(self, data, x, u):
        self.calc(data, x, u)
        self.rx[0][:19][self.ankle1_idx_qs] = self.ankleActuator1.compute_Ja()
        self.rx[1][:19][self.ankle2_idx_qs] = self.ankleActuator2.compute_Ja()
        data.Rx = self.rx
        data.Ru = np.zeros((self.nr, self.nu))

    def createData(self, collector):
        # what is collector?
        # type croc.DataCollectorAbstract
        return ResidualDataAnkleLimits(self, collector)
        #data = ResidualDataAnkleLimits(self)

class ResidualDataAnkleLimits(croc.ResidualDataAbstract):
    def __init__(self, model, data):
        super().__init__(model, data)


if __name__ == "__main__":
    pass