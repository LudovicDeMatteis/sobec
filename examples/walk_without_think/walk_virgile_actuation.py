import pinocchio as pin
import crocoddyl as croc
import numpy as np
import matplotlib.pylab as plt  # noqa: F401
from numpy.linalg import norm, pinv, inv, svd, eig  # noqa: F401

# Local imports
import sobec
import sobec.walk_without_think.plotter
import specific_params
# Since the actuation model is built from the model,
# We load the robot from this script, that calls the loader itself.
# We could save the model properties in order to have a fixed actuation model.
from sobec.walk_without_think.actuation_model import battobotAct, model, dispWithBars
from sobec.walk_without_think.actuation_model import robot as battobotRobot
# #####################################################################################
# ## TUNING ###########################################################################
# #####################################################################################

# In the code, cost terms with 0 weight are commented for reducing execution cost
# An example of working weight value is then given as comment at the end of the line.
# When setting them to >0, take care to uncomment the corresponding line.
# All these lines are marked with the tag ##0##.

# Adapted from loader virgile
from loaders_virgile import load_3d

# This robot is already freezed inside the actuation model script
robot = load_3d(battobotRobot)

walkParams = specific_params.WalkBattobotParamsActuation()

# #####################################################################################
# ### LOAD ROBOT ######################################################################
# #####################################################################################


assert len(walkParams.stateImportance) == robot.model.nv * 2

# #####################################################################################
# ### CONTACT PATTERN #################################################################
# #####################################################################################
try:
    # If possible, the initial state and contact pattern are taken from a file.
    ocpConfig = sobec.wwt.loadProblemConfig()
    contactPattern = ocpConfig["contactPattern"]
    robot.x0 = ocpConfig["x0"]
    stateTerminalTarget = ocpConfig["stateTerminalTarget"]
except (KeyError, FileNotFoundError):
    # When the config file is not found ...
    # Initial config, also used for warm start, both taken from robot wrapper.
    # Contact are specified with the order chosen in <contactIds>.
    cycle = ( [[1, 0]] * walkParams.Tsingle
              + [[1, 1]] * walkParams.Tdouble
              + [[0, 1]] * walkParams.Tsingle
              + [[1, 1]] * walkParams.Tdouble
             )
    contactPattern = (
        []
        + [[1, 1]] * walkParams.Tstart
        + (cycle * 4)
        + [[1, 1]] * walkParams.Tend
        + [[1, 1]]
    )

# #####################################################################################
# ### VIZ #############################################################################
# #####################################################################################


try:
    import meshcat
    from pinocchio.visualize import MeshcatVisualizer
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6004")
    viz.clean()
    viz.loadViewerModel(rootNodeName="universe")
except (ImportError, AttributeError):
    print("No viewer")


q0 = robot.model.referenceConfigurations["half_sitting"]
#print(
#    "Start from q0=",
#    "half_sitting"
#    if norm(q0 - robot.model.referenceConfigurations["half_sitting"]) < 1e-9
#    else q0,
#)

# #####################################################################################
# ### DDP #############################################################################
# #####################################################################################

ddp = sobec.wwt.buildSolverActuation(robot, contactPattern, walkParams)
problem = ddp.problem
x0s, u0s = sobec.wwt.buildInitialGuessActuation(ddp.problem, walkParams)
ddp.setCallbacks([croc.CallbackVerbose(), croc.CallbackLogger()])

with open("/tmp/virgile-repr.ascii", "w") as f:
    f.write(sobec.reprProblem(ddp.problem))
    print("OCP described in /tmp/virgile-repr.ascii")

croc.enable_profiler()
print("start solving")
ddp.solve(x0s, u0s, 200)
print("solved")

# assert sobec.logs.checkGitRefs(ddp.getCallbacks()[1], "refs/virgile-logs.npy")

# ### PLOT ######################################################################
# ### PLOT ######################################################################
# ### PLOT ######################################################################

sol = sobec.wwt.SolutionActuation(robot, ddp)

plotter = sobec.wwt.plotter.WalkPlotter(robot.model, robot.contactIds)
plotter.setData(contactPattern, sol.xs, sol.us, sol.fs0)

target = problem.terminalModel.differential.costs.costs[
    "stateReg"
].cost.residual.reference
forceRef = [
    sobec.wwt.plotter.getReferenceForcesFromProblemModels(problem, cid)
    for cid in robot.contactIds
]
forceRef = [np.concatenate(fs) for fs in zip(*forceRef)]

plotter.plotBasis(target)
plotter.plotTimeCop()
plotter.plotCopAndFeet(walkParams.footSize, [0,1.2,-.3,.3])
plotter.plotForces(forceRef)
plotter.plotCom(robot.com0)
plotter.plotFeet()
plotter.plotFootCollision(walkParams.footMinimalDistance)
plotter.plotJointTorques()
print("Run ```plt.ion(); plt.show()``` to display the plots.")
plt.ion(); plt.show()
# ## DEBUG ######################################################################
# ## DEBUG ######################################################################
# ## DEBUG ######################################################################

pin.SE3.__repr__ = pin.SE3.__str__
np.set_printoptions(precision=2, linewidth=300, suppress=True, threshold=10000)

while input("Press q to quit the visualisation") != "q":
    viz.play(np.array(ddp.xs)[:, : robot.model.nq], walkParams.DT)

# for x in ddp.xs:
#     viz.display(x[:robot.model.nq])
    # ims.append( viz.viewer.get_image())
# import imageio # pip install imageio[ffmpeg]
# imageio.mimsave("/tmp/battobot.mp4", imgs, 1//walkParams.DT)

# Save controls in external file
np.save("xs.npy",ddp.xs)
np.save("us.npy",ddp.us)


def play(createBarsVisuals=False):
    for x in ddp.xs:
        # print(len(x))
        #viz.display(x[: model.nq])
        x = x[: model.nq]
        dispWithBars(x[: model.nq],viz,createBarsVisuals=createBarsVisuals)
        # sleep(dt)