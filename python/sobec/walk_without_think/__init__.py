# flake8: noqa
from . import ocp, weight_share, robot_wrapper, config_mpc, save_traj, actuation_matrix
from . import miscdisp, params, yaml_params
#-Added for battobot-
from . import battobot,battobot_crocoddyl, actuation_model, ocp_actuation
# ---------------------

#-Added for battobot-
from .ocp_actuation import Solution as SolutionActuation
from .ocp_actuation import buildSolver as buildSolverActuation
from .ocp_actuation import buildInitialGuess as buildInitialGuessActuation
from .actuation_model import model as modelActuation
from .actuation_model import robot as robotActuation
from .actuation_model import dispWithBars as dispWithBarsActuation
from .battobot_crocoddyl import BattobotActuationModelMatrix

# ---------------------
from .ocp import Solution, buildSolver, buildInitialGuess
from .params import WalkParams
from .robot_wrapper import RobotWrapper
from .miscdisp import CallbackMPCWalk, dispocp
from .yaml_params import yamlReadToParams, yamlWriteParams
from .save_traj import save_traj, loadProblemConfig
from .actuation_matrix import ActuationModelMatrix

# Don't include plotter by default, it breaks bullet
# from . import plotter
# from .plotter import WalkPlotter
