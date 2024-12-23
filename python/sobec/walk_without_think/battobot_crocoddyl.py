"""
This script uses the actuation model of the battobot robot to create a crocoddyl actuation model.
"""

import crocoddyl
import numpy as np
from time import time as now

def fixNan(nparray):
    # Replace Nan by something else than 0 to improve the behavior of the solver ?
    # nparray[np.isnan(nparray)] = 0
    return nparray


class ActuationModelMatrix(crocoddyl.ActuationModelAbstract):
    def __init__(self, state, nu, battobotAct):
        """
        Croccoddyl version of the actuation model:

        """
        super(ActuationModelMatrix, self).__init__(state, nu)
        self.battobotAct = battobotAct
        self.calcCount = 0
        self.calcDiffCount = 0
        self.timeCalc = 0
        self.timeCalcDiff = 0

    def calc(self, data, x, u):
        time_a = now() 
        self.dtau_dq, self.dtau_du = self.battobotAct.J_a(x[:19], u)
        data.tau = fixNan(self.dtau_du @ u)
        time_b = now()
        self.calcCount += 1
        self.timeCalc += (time_b - time_a)

    def calcDiff(self, data, x, u):
        # Specify the actuation jacobian
        time_a = now()
        # calc should be cald each time before calcDiff.
        # It means that act_matrix is already computed in calc
        # but also that current tau_m is already computed in data.tau
        self.calc(data, x[:19], u)

        # to comment
        # dtau_dq_nd, dtau_du_nd = self.battobotAct.dtau_double_numdiff(x[:19], u)

        # wrt u
        data.dtau_du = fixNan(self.dtau_du)
        # wrt x
        # dv
        data.dtau_dx[:, 18:] = np.zeros((18, 18))
        # dq
        data.dtau_dx[:, :18] = fixNan(self.dtau_dq)
        time_b = now()
        self.calcDiffCount += 1
        self.timeCalcDiff += (time_b - time_a)
        


    def createData(self):
        data = ActuationDataMatrix(self)
        return data


class ActuationDataMatrix(crocoddyl.ActuationDataAbstract):
    def __init__(self, model):
        super(ActuationDataMatrix, self).__init__(model)


if __name__ == "__main__":
    from tests.test_actuator_numdiff import battobotAct, model

    state = crocoddyl.StateMultibody(model)
    nu = 12
    actuation = ActuationModelMatrix(state, nu, battobotAct)
    actuation_data = actuation.createData()
