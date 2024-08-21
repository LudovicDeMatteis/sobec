import pinocchio as pin
import crocoddyl as croc
import numpy as np
import matplotlib.pylab as plt  # noqa: F401
from numpy.linalg import norm, pinv, inv, svd, eig  # noqa: F401

# Local imports
import sobec
import sobec.walk_without_think.plotter
import specific_params
from loaders_virgile import load_complete_closed_6d as load_complete_closed

# #####################################################################################
# ## TUNING ###########################################################################
# #####################################################################################

# In the code, cost terms with 0 weight are commented for reducing execution cost
# An example of working weight value is then given as comment at the end of the line.
# When setting them to >0, take care to uncomment the corresponding line.
# All these lines are marked with the tag ##0##.

WS = False # Warm start

walkParams = specific_params.WalkBattobotParams(model="closed")
walkParams.saveFile = "/tmp/walk_virgile_closed.npy"
if WS:
    walkParams.guessFile = "/tmp/walk_virgile_closed_ws.npy"
# #####################################################################################
# ### LOAD ROBOT ######################################################################
# #####################################################################################

robot = load_complete_closed()
assert len(walkParams.stateImportance) == robot.model.nv * 2
assert len(walkParams.stateTerminalImportance) == robot.model.nv * 2

robot.loop_constraints_models = []
robot.actuationModel = None

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
        + (cycle * 2)
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
ddp.solve(x0s, u0s, 1)

# assert sobec.logs.checkGitRefs(ddp.getCallbacks()[1], "refs/virgile-logs.npy")

# ### PLOT ######################################################################
# ### PLOT ######################################################################
# ### PLOT ######################################################################

sol = sobec.wwt.Solution(robot, ddp)

# for cost_name in ddp.problem.runningModels[0].differential.costs.costs.todict().keys():
#     for t in [0, 10, 20, 30, 40, 50]:
#         if cost_name not in ddp.problem.runningModels[t].differential.costs.costs.todict().keys():
#             continue
#         x = np.array(sol.xs.tolist())[t]
#         q = x[:robot.model.nq]
#         v = x[robot.model.nq :]
#         u = np.array(sol.us.tolist())[t]
#         dam = ddp.problem.runningModels[t].differential
#         dam_data = dam.createData()
#         dam.calc(dam_data, x, u)
#         dam.calcDiff(dam_data, x, u)

#         # r0 = dam_data.xout.copy() #
#         r0 = dam_data.costs.costs.todict()[cost_name].residual.r.copy()
#         Rx_gt = dam_data.costs.costs.todict()[cost_name].residual.Rx
#         Ru_gt = dam_data.costs.costs.todict()[cost_name].residual.Ru

#         Rx_fd = np.empty((r0.size, v.size * 2))
#         Ru_fd = np.empty((r0.size, u.size))
#         eps = 1e-7
#         q_eps = np.zeros(v.size)
#         for i in range(v.size):
#             q_eps[i] += eps
#             q1 = pin.integrate(robot.model, q, q_eps)
#             x1 = np.concatenate([q1, v])
#             dam.calc(dam_data, x1, u)
#             r1 = dam_data.costs.costs.todict()[cost_name].residual.r.copy()
#             Rx_fd[:, i] = (r1 - r0) / eps
#             q_eps[i] -= eps
#         v1 = v.copy()
#         for i in range(v.size):
#             v1[i] += eps
#             x1 = np.concatenate([q, v1])
#             dam.calc(dam_data, x1, u)
#             r1 = dam_data.costs.costs.todict()[cost_name].residual.r
#             Rx_fd[:, i + v.size] = (r1 - r0) / eps
#             v1[i] -= eps
#         u1 = u.copy()
#         for i in range(u.size):
#             u1[i] += eps
#             dam.calc(dam_data, x, u1)
#             r1 = dam_data.costs.costs.todict()[cost_name].residual.r
#             Ru_fd[:, i] = (r1 - r0) / eps
#             u1[i] -= eps
        # np.testing.assert_allclose(Rx_fd, Rx_gt, atol=2*np.sqrt(eps))
        # np.testing.assert_allclose(Ru_fd, Ru_gt, atol=2*np.sqrt(eps))
        # print(f"t={t}", "OK for", cost_name)
x = np.array(sol.xs.tolist())[44]
q = x[:robot.model.nq]
v = x[robot.model.nq :]
u = np.array(sol.us.tolist())[44]
act_matrix = np.zeros((robot.model.nv, len()))
for iu, iv in enumerate(robot.actuationModel.mot_ids_v):
    act_matrix[iv, iu] = 1
tau = act_matrix @ u
## Check the dynamic derivative
# Create the constraints
loop_constraints_models = robot.loop_constraints_models
floor_constraints_models = []
for cid in robot.contactIds:
    cstr = pin.RigidConstraintModel(
        pin.ContactType.CONTACT_6D,
        robot.model,
        robot.model.frames[cid].parentJoint,
        robot.model.frames[cid].placement,
        pin.ReferenceFrame.LOCAL,
    )
    floor_constraints_models.append(cstr)
# Create the data
loop_constraints_data = [cstr.createData() for cstr in loop_constraints_models]
floor_constraints_data = [cstr.createData() for cstr in floor_constraints_models]
data = robot.model.createData()
# Compute the dynamics
pin.initConstraintDynamics(robot.model, data, loop_constraints_models + floor_constraints_models)
acc = pin.constraintDynamics(robot.model, data, q, v, tau, loop_constraints_models + floor_constraints_models, loop_constraints_data + floor_constraints_data)
# * Compute the pinocchio derivative
pin.computeConstraintDynamicsDerivatives(robot.model, data, loop_constraints_models + floor_constraints_models, loop_constraints_data + floor_constraints_data)
# * Compute finite Differences
# r0 = dam_data.xout.copy() #
acc0 = acc.copy()
Fq_gt = data.ddq_dq
Fv_gt = data.ddq_dv
Ftau_gt = data.ddq_dtau

Fq_fd = np.empty((acc0.size, v.size))
Fv_fd = np.empty((acc0.size, v.size))
Ftau_fd = np.empty((acc0.size, tau.size))
eps = 1e-7
q_eps = np.zeros(v.size)
for i in range(v.size):
    q_eps[i] += eps
    q1 = pin.integrate(robot.model, q, q_eps)
    acc = pin.constraintDynamics(robot.model, data, q1, v, tau, loop_constraints_models + floor_constraints_models, loop_constraints_data + floor_constraints_data)
    acc1 = acc.copy()
    Fq_fd[:, i] = (acc1 - acc0) / eps
    q_eps[i] -= eps
v1 = v.copy()
for i in range(v.size):
    v1[i] += eps
    acc = pin.constraintDynamics(robot.model, data, q, v1, tau, loop_constraints_models + floor_constraints_models, loop_constraints_data + floor_constraints_data)
    acc1 = acc.copy()
    Fv_fd[:, i] = (acc1 - acc0) / eps
    v1[i] -= eps
tau1 = tau.copy()
for i in range(tau.size):
    tau1[i] += eps
    acc = pin.constraintDynamics(robot.model, data, q, v, tau1, loop_constraints_models + floor_constraints_models, loop_constraints_data + floor_constraints_data)
    acc1 = acc.copy()
    Ftau_fd[:, i] = (acc1 - acc0) / eps
    tau1[i] -= eps

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

for t in range(20, 50):
    print(t)
    x = ddp.xs[t]
    u = ddp.us[t]
    data = robot.model.createData()
    pin.centerOfMass(robot.model, data, x[:q.size], x[q.size:])
    print(data.vcom[0])

    cd = ddp.problem.runningDatas[t].differential.costs.costs.todict()["comVelCost"].residual
    print(cd.r)
    cm = ddp.problem.runningModels[t].differential.costs.costs.todict()["comVelCost"].cost.residual
    cm.calc(cd, x, u)
    print(cd.r)

    cd2 = cm.createData(ddp.problem.runningDatas[t].differential.multibody)
    pin.centerOfMass(robot.model, cd2.pinocchio, x[:q.size], x[q.size:]) # This line correct the COM position (without it it looks wrong)
    cm.calc(cd2, x, u)
    print(cd2.r)


pin.SE3.__repr__ = pin.SE3.__str__
np.set_printoptions(precision=2, linewidth=300, suppress=True, threshold=10000)

while input("Press q to quit the visualisation") != "q":
    viz.play(np.array(ddp.xs)[:, : robot.model.nq], walkParams.DT)

if walkParams.saveFile is not None and input("Save trajectory? (y/n)") == "y":
    sobec.wwt.save_traj(np.array(sol.xs), filename=walkParams.saveFile)

# imgs = []
# import time
# for x in ddp.xs:
#     viz.display(x[:robot.model.nq])
#     time.sleep(0.05)
#     imgs.append( viz.viewer.get_image())
# import imageio # pip install imageio[ffmpeg]
# imageio.mimsave("/tmp/battobot.mp4", imgs, fps=1//walkParams.DT)
