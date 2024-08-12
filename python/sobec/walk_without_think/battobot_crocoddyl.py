
######
# This script uses the Jacobians and their derivatives to build the croccodyl actuation model.
######


import pinocchio as pin
import numpy as np
import numdifftools as nd
from numpy.testing import assert_almost_equal
import crocoddyl



class BattobotActuationModelMatrix(crocoddyl.ActuationModelAbstract):
    def __init__(
        self, state, nu, battobotAct
    ):
        """
        Croccoddyl version of the actuation model:

        """
        super(BattobotActuationModelMatrix, self).__init__(state, nu)
        self.battobotAct = battobotAct

    def calc(self, data, x, u):
        
        self.dtau_dq,self.dtau_du = self.battobotAct.J_a(x[:19],u)
        np.printoptions(linewidth=550,precision=3)
        #print("self.dtau_dq",self.dtau_dq)
        #print("self.dtau_du",self.dtau_du)
        data.tau = self.dtau_du @ u  

    def calcDiff(self, data, x, u):
        # Specify the actuation jacobian

        # calc should be cald each time before calcDiff.
        # It means that act_matrix is already computed in calc
        # but also that current tau_m is already computed in data.tau
        self.calc(data, x[:19], u)

        # wrt u
        data.dtau_du = self.dtau_du

        # wrt x
        # dv
        data.dtau_dx[:,18:] = np.zeros((18,18))
        # dq
        data.dtau_dx[:,:18] = self.dtau_dq

        #print("data.dtau_dx",data.dtau_dx)
    
    def createData(self):
        data = BattobotActuationDataMatrix(self)
        return data


class BattobotActuationDataMatrix(crocoddyl.ActuationDataAbstract):
    def __init__(self, model):
        super(BattobotActuationDataMatrix, self).__init__(model)


if __name__ == "__main__":
    
    print("Tests unimplemented in sobec.")
    #from test_actuator import battobotAct,model
    #state = crocoddyl.StateMultibody(model)
    #nu = 12
    #actuation = ActuationModelMatrix(state, nu, battobotAct)
    #actuation_data = actuation.createData()