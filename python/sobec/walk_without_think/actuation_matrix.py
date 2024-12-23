import crocoddyl
import numpy as np
import sobec
from time import time as now

class ActuationModelMatrix(crocoddyl.ActuationModelAbstract):
    def __init__(self, state, nu, act_matrix):
        super(ActuationModelMatrix, self).__init__(state, nu)
        self.ntau = state.nv
        assert(act_matrix.shape[0] == self.ntau)
        assert(act_matrix.shape[1] == nu)

        self.act_matrix = act_matrix
        self.calcCount = 0
        self.calcDiffCount = 0
        self.timeCalc = 0
        self.timeCalcDiff = 0

    def calc(self, data, x, u):
        time_a = now()
        data.tau = self.act_matrix @ u
        time_b = now()
        self.timeCalc += (time_b - time_a)
        self.calcCount +=1

    def calcDiff(self, data, x, u):
        # Specify the actuation jacobian
        time_a = now()
        # wrt u
        data.dtau_du = self.act_matrix # np.transpose(act_matrix)

        # wrt x
        data.dtau_dx[:, :] = np.zeros((self.ntau, self.state.ndx))
        time_b = now()
        self.timeCalcDiff += (time_b - time_a)
        self.calcDiffCount +=1

    def createData(self):
        data = ActuationDataMatrix(self)
        return data
    
class ActuationDataMatrix(crocoddyl.ActuationDataAbstract):
    def __init__(self, model):
        super(ActuationDataMatrix, self).__init__(model)
