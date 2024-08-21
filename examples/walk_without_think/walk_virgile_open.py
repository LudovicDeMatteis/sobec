import pinocchio as pin
import crocoddyl as croc
import numpy as np
import matplotlib.pylab as plt  # noqa: F401
from numpy.linalg import norm, pinv, inv, svd, eig  # noqa: F401

# Local imports
import sobec
import sobec.walk_without_think.plotter
import specific_params
from loaders_virgile import load_complete_open

# #####################################################################################
# ## TUNING ###########################################################################
# #####################################################################################

# In the code, cost terms with 0 weight are commented for reducing execution cost
# An example of working weight value is then given as comment at the end of the line.
# When setting them to >0, take care to uncomment the corresponding line.
# All these lines are marked with the tag ##0##.

walkParams = specific_params.WalkBattobotParams()
walkParams.saveFile = "/tmp/walk_virgile_open.npy"

# #####################################################################################
# ### LOAD ROBOT ######################################################################
# #####################################################################################

robot = load_complete_open()
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
    viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    viz.clean()
    viz.loadViewerModel(rootNodeName="universe")
except (ImportError, AttributeError):
    print("No viewer")


q0 = robot.x0[: robot.model.nq]
print(
    "Start from q0=",
    "half_sitting"
    if norm(q0 - robot.model.referenceConfigurations["half_sitting"]) < 1e-9
    else q0,
)

# #####################################################################################
# ### DDP #############################################################################
# #####################################################################################

ddp = sobec.wwt.buildSolver(robot, contactPattern, walkParams, solver='FDDP')
problem = ddp.problem
x0s, u0s = sobec.wwt.buildInitialGuess(ddp.problem, walkParams)
ddp.setCallbacks([croc.CallbackVerbose(), croc.CallbackLogger()])

with open("/tmp/virgile-repr.ascii", "w") as f:
    f.write(sobec.reprProblem(ddp.problem))
    print("OCP described in /tmp/virgile-repr.ascii")

croc.enable_profiler()
ddp.solve(x0s, u0s, 200)

# assert sobec.logs.checkGitRefs(ddp.getCallbacks()[1], "refs/virgile-logs.npy")
# ### PLOT ######################################################################
# ### PLOT ######################################################################
# ### PLOT ######################################################################

sol = sobec.wwt.Solution(robot, ddp)

### VERIFY COSTS VALUES AND DERIVATIVES ########################################
for cost_name in ddp.problem.runningModels[0].differential.costs.costs.todict().keys():
    for t in [0, 10, 20, 30, 40, 50]:
        if cost_name not in ddp.problem.runningModels[t].differential.costs.costs.todict().keys():
            continue
        x = np.array(sol.xs.tolist())[t]
        q = x[:robot.model.nq]
        v = x[robot.model.nq :]
        u = np.array(sol.us.tolist())[t]
        dam = ddp.problem.runningModels[t].differential
        dam_data = dam.createData()
        dam.calc(dam_data, x, u)
        dam.calcDiff(dam_data, x, u)

        # r0 = dam_data.xout.copy() #
        r0 = dam_data.costs.costs.todict()[cost_name].residual.r.copy()
        Rx_gt = dam_data.costs.costs.todict()[cost_name].residual.Rx
        Ru_gt = dam_data.costs.costs.todict()[cost_name].residual.Ru

        Rx_fd = np.empty((r0.size, v.size * 2))
        Ru_fd = np.empty((r0.size, u.size))
        eps = 1e-8
        q_eps = np.zeros(v.size)
        for i in range(v.size):
            q_eps[i] += eps
            q1 = pin.integrate(robot.model, q, q_eps)
            x1 = np.concatenate([q1, v])
            dam.calc(dam_data, x1, u)
            r1 = dam_data.costs.costs.todict()[cost_name].residual.r.copy()
            Rx_fd[:, i] = (r1 - r0) / eps
            q_eps[i] -= eps
        v1 = v.copy()
        for i in range(v.size):
            v1[i] += eps
            x1 = np.concatenate([q, v1])
            dam.calc(dam_data, x1, u)
            r1 = dam_data.costs.costs.todict()[cost_name].residual.r
            Rx_fd[:, i + v.size] = (r1 - r0) / eps
            v1[i] -= eps
        u1 = u.copy()
        for i in range(u.size):
            u1[i] += eps
            dam.calc(dam_data, x, u1)
            r1 = dam_data.costs.costs.todict()[cost_name].residual.r
            Ru_fd[:, i] = (r1 - r0) / eps
            u1[i] -= eps
        np.testing.assert_allclose(Rx_fd, Rx_gt, atol=2*np.sqrt(eps))
        np.testing.assert_allclose(Ru_fd, Ru_gt, atol=2*np.sqrt(eps))
        print(f"t={t}", "OK for", cost_name)

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
plotter.plotCopAndFeet(walkParams.footSize, 0.6)
plotter.plotForces(forceRef)
plotter.plotCom(robot.com0)
plotter.plotFeet()
plotter.plotFootCollision(walkParams.footMinimalDistance)
print("Run ```plt.ion(); plt.show()``` to display the plots.")
# plt.ion()
# plt.show()

costPlotter = sobec.wwt.plotter.CostPlotter(robot.model, ddp)
costPlotter.setData()
costPlotter.plotCosts()
plt.show()

from matplotlib.backends.backend_pdf import PdfPages
def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
multipage("figs.pdf", [plt.figure(i) for i in plt.get_fignums()])

# ## DEBUG ######################################################################
# ## DEBUG ######################################################################
# ## DEBUG ######################################################################

pin.SE3.__repr__ = pin.SE3.__str__
np.set_printoptions(precision=2, linewidth=300, suppress=True, threshold=10000)

while input("Press q to quit the visualisation") != "q":
    viz.play(np.array(ddp.xs)[:, : robot.model.nq], walkParams.DT)

if walkParams.saveFile is not None and input("Save trajectory? (y/n)") == "y":
    sobec.wwt.save_traj(np.array(sol.xs), np.array(sol.us), filename=walkParams.saveFile)

# us = np.array(ddp.us.tolist())
# plt.figure()
# plt.plot(us[:, [0, 1, 2, 3]])
# plt.show()

# for x in ddp.xs:
#     viz.display(x[:robot.model.nq])
    # ims.append( viz.viewer.get_image())
# import imageio # pip install imageio[ffmpeg]
# imageio.mimsave("/tmp/battobot.mp4", imgs, 1//walkParams.DT)
