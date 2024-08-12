
######
# This script computes the actuation model (Jacobians) for the knee and ankle
# (and then full battobot) 
######

import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_almost_equal
import numdifftools as nd  # pip install numdifftools
import matplotlib.pyplot as plt
import pinocchio as pin
from pinocchio.utils import rotate
from .battobot import loadBattobot, freezeSide, freezeActuation, alignModel
from .battobot import kneeGeomARight, ankle1GeomARight, ankle2GeomARight
from .battobot import kneeGeomALeft, ankle1GeomALeft, ankle2GeomALeft
from .battobot import (
    kneeFramesRight,
    kneeParentRight,
    ankle1FramesRight,
    ankle1ParentRight,
    ankle1GeomRight,
    ankle2FramesRight,
    ankle2ParentRight,
    ankle2GeomRight,
    kneeOutputRight,
    ankle1OutputRight,
    ankle2OutputRight,
)
from .battobot import (
    kneeFramesLeft,
    kneeParentLeft,
    ankle1FramesLeft,
    ankle1ParentLeft,
    ankle1GeomLeft,
    ankle2FramesLeft,
    ankle2ParentLeft,
    ankle2GeomLeft,
    kneeOutputLeft,
    ankle1OutputLeft,
    ankle2OutputLeft,
)

#from meshcat_viewer_wrapper.visualizer import MeshcatVisualizer
import pinocchio as pin
import numpy as np
#import sliders


sumsqr = lambda v: sum(v**2)
sqrt = np.sqrt
arccos, arcsin, arctan, arctan2 = np.arccos, np.arcsin, np.arctan, np.arctan2
A_ = np.array
cos, sin = np.cos, np.sin
array = np.array


robot = freezeActuation(loadBattobot(withFreeFlyer=True))
model = robot.model
alignModel(model, robot.q0)
data = model.createData()
robot.rebuildData()

pin.framesForwardKinematics(model, data, robot.q0)


def universalJointModel(l3, l4):
    if not isinstance(l3, np.ndarray):
        l3 = np.array([l3, 0, 0])
    if not isinstance(l4, np.ndarray):
        l4 = np.array([l4, 0, 0])

    model = pin.Model()
    jointId = 0

    jointId = model.addJoint(jointId, pin.JointModelRZ(), pin.SE3(np.eye(3), l4), "RZ")

    jointId = model.addJoint(
        jointId, pin.JointModelRY(), pin.SE3(np.eye(3), np.zeros(3)), "RY"
    )

    model.addFrame(pin.Frame("m", 0, 0, pin.SE3.Identity(), pin.FrameType.OP_FRAME))
    model.addFrame(
        pin.Frame("b", jointId, jointId, pin.SE3(np.eye(3), l3), pin.FrameType.OP_FRAME)
    )
    return model


class FourBarsActuator:
    def __init__(self, model=None):
        if model is None:
            self.l1, self.l2, self.l4 = 0.8, 1.6, 1.1
            self.l3 = A_([0.1, 0.05, 0.02])
            self.s_init = A_([0.1, 0.2])

            self.model = universalJointModel(self.l3, self.l4)
            self.withSanityCheck = True
            self.setSerialJoints(0, 2)
            self.setMotor(self.model.getFrameId("m"), self.model.getFrameId("b"))
        else:
            self.model = model
            self.withSanityCheck = False
        self.data = self.model.createData()

    def setMotorArmLength(self, l1):
        """
        Length of the bar attached to the motor (i.e. MA on our drawings)
        """
        self.l1 = l1

    def setBindingArmLength(self, l2):
        """
        Length of the bar attached to the (free) bindings (i.e. AB on our drawings)
        """
        self.l2 = l2

    def setSerialJoints(self, parentId, endId):
        assert parentId < len(self.model.joints)
        assert endId < len(self.model.joints)
        self.idj_parent = parentId
        self.idj_end = j = endId
        self.serialJoints = []
        while True:
            assert self.model.nvs[j] == 1
            self.serialJoints.insert(0, j)
            j = self.model.parents[j]
            if j == self.idj_parent:
                break
            assert j > 0

        self.idx_qs = [self.model.idx_qs[i] for i in self.serialJoints]
        self.idx_vs = [self.model.idx_vs[i] for i in self.serialJoints]

    def setMotor(self, motorFrameId, bindingFrameId):
        assert motorFrameId < len(self.model.frames)
        assert bindingFrameId < len(self.model.frames)
        self.idf_m = motorFrameId
        self.idf_b = bindingFrameId

    def fk(self, s):
        """
        Compute the world position of the linkages, and the corresponding derivative wrt s
        That should be evaluated in motor frame ... TODO
        For convenience, return the pairs (b,Bs)
        """
        pin.framesForwardKinematics(self.model, self.data, s)
        self.inm_M0 = self.data.oMf[self.idf_m].inverse()
        self.b = self.inm_M0 * self.data.oMf[self.idf_b].translation

        pin.computeJointJacobians(self.model, self.data, s)
        self.Bs = self.inm_M0.rotation @ self.data.J[:3, self.idx_vs]
        self.Bs -= (
            pin.skew(self.b - self.inm_M0.translation)
            @ self.inm_M0.rotation
            @ self.data.J[3:, self.idx_vs]
        )

        if self.withSanityCheck:
            self.kineCheck(s)  # Sanity check
        return self.b, self.Bs

    def actuation(self, s, u):
        l1, l2 = self.l1, self.l2
        # Compute the angle m
        b, Bs = self.fk(s)

        l__2 = self.l__2 = sumsqr(b[:2])
        l = self.l = np.sqrt(l__2)
        l2b__2 = self.l2b__2 = l2**2 - b[2] ** 2
        l2b = self.l2b = sqrt(l2b__2)

        rc = self.rc = (l__2 + l1**2 - l2b__2) / (2 * l * l1)
        rs = self.rs = sqrt(1 - rc**2)

        m1 = arctan2(b[1], b[0])
        m2 = arccos(rc)
        m = self.m = np.r_[m1 + m2]

        # Compute the jacobian
        mu = (rc * l1 - l) / (rs * l__2 * l1)
        nu = 1 / l__2
        xi = -1 / (rs * l * l1)
        K = self.K = A_([[mu, nu, 0], [-nu, mu, 0], [0, 0, xi]])

        # print("K.T shape", (K.T).shape)
        # print("b shape", b[:,np.newaxis].shape)
        # print("u shape", u.shape,u)
        self.inm_fb = K.T @ (b[:, np.newaxis] @ u)
        self.taus = Bs.T @ self.inm_fb

        return self.taus

    def actuationDiff(self, s, u):
        # Bs.T @ K.T @ b @ u
        self.actuation(s, u)

        l1, b, l__2, rc, rs = self.l1, self.b, self.l__2, self.rc, self.rs
        Bs, l, K = self.Bs, self.l, self.K
        mu, xi = self.K[0, 0], self.K[2, 2]
        rc, rs = self.rc, self.rs

        # ### Compute df/db
        dmu_dm2 = (-l1 + rc * l) / (rs**2 * l__2 * l1)
        dmu_dl = (l - 2 * rc * l1) / (rs * l**3 * l1)
        dnu_dl = -2 / l**3
        dxi_dm2 = -rc / rs * xi
        dxi_dl = -xi / l
        dm2_db = [mu, mu, xi] * b
        Km2_times_b = [dmu_dm2, dmu_dm2, dxi_dm2] * b
        Kl = A_([[dmu_dl, dnu_dl, 0], [-dnu_dl, dmu_dl, 0], [0, 0, dxi_dl]])
        # Tried to optimze a bit the next line of code. Not excellent ...
        # dfb_db = u[0]*( K.T + Km2.T@bxyz@dm2_db + Kl.T@bxyz@bxy.T/l)
        dfb_db = K.T + Km2_times_b[:, np.newaxis] @ dm2_db[np.newaxis, :]
        dfb_db[:, :2] += (Kl.T @ b)[:, np.newaxis] @ b[np.newaxis, :2] / l
        dfb_db *= u[0]

        # ### Compute d(Bs.T fb)/ds
        rm, rd = self.model, self.data
        inm_b = self.b
        inm_fb = self.inm_fb
        inm_phi = pin.Force(inm_fb, np.cross(inm_b, inm_fb))
        in0_J = rd.J
        inm_J = self.inm_M0.action @ rd.J[:, self.idx_vs]

        # Compute the derivative of the force transmission
        # Bss.fb = d_BsTfb_ds = df1 + df2
        # with df1 = J.T@dphi_ds the part of the derivative due to the torque change in s
        # with df2 the part of the derivative due to the serial-joint hessian
        # Compute df1
        dphi_ds = np.c_[-pin.skew(inm_fb), pin.skew(inm_fb) @ pin.skew(inm_b)] @ inm_J
        d_BsTfb_ds = inm_J.T[:, 3:] @ dphi_ds
        # Compute df2
        # The computation below is super nice, but finally can be much simpler!
        # Skip the computation of FY[0] which leads to a trivial 0 result when multiply with J
        if len(self.serialJoints) == 2:
            iv0, iv1 = self.idx_vs
            inm_F = inm_J[:, 1]
            FY = np.r_[
                np.cross(inm_F[3:], inm_phi.linear),
                np.cross(inm_F[:3], inm_phi.linear)
                + np.cross(inm_F[3:], inm_phi.angular),
            ]
            d_BsTfb_ds[1, 0] += FY @ inm_J[:, 0]
        else:
            FY = [[]]
            for k in range(1, len(self.serialJoints)):
                assert self.model.nvs[self.serialJoints[k]] == 1
                inm_F = inm_J[:, k]
                FY.append(
                    np.r_[
                        np.cross(inm_F[3:], inm_phi.linear),
                        np.cross(inm_F[:3], inm_phi.linear)
                        + np.cross(inm_F[3:], inm_phi.angular),
                    ]
                )
            for r in range(1, len(self.serialJoints)):
                for c in range(r):
                    d_BsTfb_ds[r, c] += FY[r] @ inm_J[:, c]

        # Finally, assemble that mess
        dtaus_ds = d_BsTfb_ds + Bs.T @ dfb_db @ Bs
        return dtaus_ds

    # ### Checks for unittest
    def kineCheck(self, s):
        """
        for the particular case of l3,l4, check the kinematics
        """
        # Check that the computation of b is the same with pinocchio
        b = [self.l4, 0, 0] + rotate("z", s[0]) @ rotate("y", s[1]) @ self.l3
        assert_almost_equal(self.b, b)

        # Check that the computation is the same with pinocchio
        l3, l4 = self.l3, self.l4
        M0 = pin.SE3(np.eye(3), A_([l4, 0, 0]))
        M1 = pin.SE3(rotate("z", s[0]), np.zeros(3))
        M2 = pin.SE3(rotate("y", s[1]), np.zeros(3))
        J = np.c_[
            M0.action @ A_([0, 0, 0, 0, 0, 1]),
            (M0 * M1).action @ A_([0, 0, 0, 0, 1, 0]),
        ]
        df = J[:3] - pin.skew(b) @ J[3:]
        assert_almost_equal(df, self.Bs)

    def compute_Ja(self):
        Ja = self.Ja = self.b[np.newaxis, :] @ self.K @ self.Bs
        return Ja

    def compute_Ja_numdiff(self, s):
        def s_to_m(s):
            self.actuation(s[:, 0], np.array([0]))
            return self.m

        return nd.Jacobian(s_to_m)(s[:, np.newaxis])

    def actuationNumdiff(self, s, u):
        As_ = nd.Jacobian(lambda s: self.actuation(s, u))
        As = As_(s)
        return As

    def computeMidPoints(self):
        """
        For debug mostly: evaluate the position of point A and E
        """
        l1, l2, b, l__2, rs, rc = self.l1, self.l2, self.b, self.l__2, self.rs, self.rc
        R = A_([[rc, -rs], [rs, rc]])
        self.e = R @ b[:2]
        self.a = self.e * l1 / np.sqrt(l__2)

    def assertValues(self):
        l1, l2, b, l__2, rs, rc = self.l1, self.l2, self.b, self.l__2, self.rs, self.rc
        self.computeMidPoints()
        a, e = self.a, self.e

        m_from_a = arctan2(a[1], a[0])

        assert_almost_equal(norm(self.a), self.l1, 1e-6)
        assert_almost_equal(norm(self.a - self.b[:2]), self.l2b, 1e-6)
        assert_almost_equal(norm(np.r_[a, 0] - self.b), self.l2, 1e-6)
        # s = A_([ self.l4,0,0 ])
        s = self.data.oMi[self.idj_end].translation
        assert_almost_equal(norm(self.b - s), self.l3, 1e-6)
        assert_almost_equal(norm(s), self.l4, 1e-6)
        assert_almost_equal(self.m, m_from_a)

    def inject_attributes_to_locals(self, loc):
        """
        Dirty hack for debug, do not use if you are not me
        """
        for attr in dir(self):
            if not attr.startswith("__") and not callable(getattr(self, attr)):
                loc[attr] = getattr(self, attr)


class MultiFourBarActuator:
    def __init__(self, model):
        self.model = model
        self.fourbars = []

    def setSerialJoints(self, parent, end):
        self.parent = parent

    def addMotor(self, idf_motor, idf_binding, topBarLength, lowerBarLength):
        pass


def dispFrames(viz):
    """
    Simple debug routine to visualize one by one all the robot frames.
    """
    for i, f in enumerate(model.frames):
        print(f"\n\n-----\n{f.name} ... ({model.names[f.parent]})")
        viz.applyConfiguration("/f", data.oMf[i])
        input()


def dispkneeFramesRight(viz):
    """
    Simple debug routine to visualize one by one the placement of the
    frame of the knee actuation.
    """
    for n in kneeFramesRight:
        i = model.getFrameId(n)
        print("\n\n---")
        print(n, i)
        print(data.oMf[i])
        viz.applyConfiguration("/f", data.oMf[i])
        input()


def dispkneeFramesLeft(viz):
    """
    Simple debug routine to visualize one by one the placement of the
    frame of the knee actuation.
    """
    for n in kneeFramesLeft:
        i = model.getFrameId(n)
        print("\n\n---")
        print(n, i)
        print(data.oMf[i])
        viz.applyConfiguration("/f", data.oMf[i])
        input()


def dispankle1FramesRight(viz):
    """
    Simple debug routine to visualize one by one the placement of the
    frame of the ankle1 actuation.
    """
    for n in ankle1FramesRight:
        i = model.getFrameId(n)
        print("\n\n---")
        print(n, i)
        print(data.oMf[i])
        viz.applyConfiguration("/f", data.oMf[i])
        input()


def dispankle2FramesRight(viz):
    """
    Simple debug routine to visualize one by one the placement of the
    frame of the ankle2 actuation.
    """
    for n in ankle2FramesRight:
        i = model.getFrameId(n)
        print("\n\n---")
        print(n, i)
        print(data.oMf[i])
        viz.applyConfiguration("/f", data.oMf[i])
        input()


def dispankle1FramesLeft(viz):
    """
    Simple debug routine to visualize one by one the placement of the
    frame of the ankle1 actuation.
    """
    for n in ankle1FramesLeft:
        i = model.getFrameId(n)
        print("\n\n---")
        print(n, i)
        print(data.oMf[i])
        viz.applyConfiguration("/f", data.oMf[i])
        input()


def dispankle2FramesLeft(viz):
    """
    Simple debug routine to visualize one by one the placement of the
    frame of the ankle2 actuation.
    """
    for n in ankle2FramesRight:
        i = model.getFrameId(n)
        print("\n\n---")
        print(n, i)
        print(data.oMf[i])
        viz.applyConfiguration("/f", data.oMf[i])
        input()


# ### ACTUATION MODELS
# Create the actuation models for right knee, ankle 1 and 2

# --- RIGHT KNEE
idf_m = model.getFrameId(kneeFramesRight[0])
idf_a = model.getFrameId(kneeFramesRight[1])
idf_bup = model.getFrameId(kneeFramesRight[2])
idf_b = model.getFrameId(kneeFramesRight[3])
kneeRight = FourBarsActuator(model)
kneeRight.data = data
kneeRight.setSerialJoints(
    model.getJointId(kneeParentRight), model.getJointId(kneeOutputRight)
)
kneeRight.setMotor(idf_m, idf_b)

OM = data.oMf[idf_m].translation
OA = data.oMf[idf_a].translation
OB_up = data.oMf[idf_bup].translation
OB_down = data.oMf[idf_b].translation
assert np.isclose(0, (data.oMf[idf_m].inverse() * OB_down)[2])
MA = np.linalg.norm(OM[[0, 2]] - OA[[0, 2]])
AB = np.linalg.norm(OA[[0, 2]] - OB_up[[0, 2]])
kneeRight.setMotorArmLength(MA)
kneeRight.setBindingArmLength(AB)


def dispkneeRightBar(kneeRight, viz):
    R = np.array([[kneeRight.rc, -kneeRight.rs], [kneeRight.rs, kneeRight.rc]])
    e = R @ kneeRight.b[:2]
    inm_a = np.r_[e * kneeRight.l1 / np.sqrt(kneeRight.l__2), 0]
    in0_a = data.oMf[kneeRight.idf_m] * inm_a
    in0_b = data.oMf[kneeRight.idf_m] * kneeRight.b
    in0_Rbar = pin.Quaternion.FromTwoVectors(
        np.array([0, 0, kneeRight.l2]), in0_b - in0_a
    ).matrix()
    oMbar = pin.SE3(in0_Rbar, (in0_a + in0_b) / 2)
    viz.applyConfiguration("kneeRightbar", oMbar)

    oMrod = pin.SE3(in0_Rbar, in0_a) * pin.SE3(np.eye(3), np.array([0, 0.02, 0]))
    viz.applyConfiguration("pinocchio/visuals/knee_rod_2_0", oMrod)

    viz.applyConfiguration(
        "kneeRight_a", pin.SE3(data.oMf[kneeRight.idf_m].rotation, in0_a)
    )
    viz.applyConfiguration(
        "kneeRight_b", pin.SE3(data.oMf[kneeRight.idf_b].rotation, in0_b)
    )


# --- LEFT KNEE

idf_m = model.getFrameId(kneeFramesLeft[0])
idf_a = model.getFrameId(kneeFramesLeft[1])
idf_bup = model.getFrameId(kneeFramesLeft[2])
idf_b = model.getFrameId(kneeFramesLeft[3])
kneeLeft = FourBarsActuator(model)
kneeLeft.data = data
kneeLeft.setSerialJoints(
    model.getJointId(kneeParentLeft), model.getJointId(kneeOutputLeft)
)
kneeLeft.setMotor(idf_m, idf_b)

OM = data.oMf[idf_m].translation
OA = data.oMf[idf_a].translation
OB_up = data.oMf[idf_bup].translation
OB_down = data.oMf[idf_b].translation
assert np.isclose(0, (data.oMf[idf_m].inverse() * OB_down)[2])
MA = np.linalg.norm(OM[[0, 2]] - OA[[0, 2]])
AB = np.linalg.norm(OA[[0, 2]] - OB_up[[0, 2]])
kneeLeft.setMotorArmLength(MA)
kneeLeft.setBindingArmLength(AB)


def dispkneeLeftBar(kneeLeft, viz):
    # Inverse here, maybe not necessary
    R = np.linalg.inv(
        np.array([[kneeLeft.rc, -kneeLeft.rs], [kneeLeft.rs, kneeLeft.rc]])
    )
    e = R @ kneeLeft.b[:2]
    inm_a = np.r_[e * kneeLeft.l1 / np.sqrt(kneeLeft.l__2), 0]
    in0_a = data.oMf[kneeLeft.idf_m] * inm_a
    in0_b = data.oMf[kneeLeft.idf_m] * kneeLeft.b
    in0_Rbar = pin.Quaternion.FromTwoVectors(
        np.array([0, 0, kneeLeft.l2]), in0_b - in0_a
    ).matrix()
    oMbar = pin.SE3(in0_Rbar, (in0_a + in0_b) / 2)
    viz.applyConfiguration("kneeLeftbar", oMbar)

    oMrod = pin.SE3(in0_Rbar, in0_a) * pin.SE3(np.eye(3), np.array([0, 0.02, 0]))
    viz.applyConfiguration("pinocchio/visuals/knee_rod_0", oMrod)

    viz.applyConfiguration(
        "kneeLeft_a", pin.SE3(data.oMf[kneeLeft.idf_m].rotation, in0_a)
    )
    viz.applyConfiguration(
        "kneeLeft_b", pin.SE3(data.oMf[kneeLeft.idf_b].rotation, in0_b)
    )


# --- RIGHT ANKLE 1
idf_m = model.getFrameId(ankle1FramesRight[0])
idf_a = model.getFrameId(ankle1FramesRight[1])
idf_bup = model.getFrameId(ankle1FramesRight[2])
idf_b = model.getFrameId(ankle1FramesRight[3])
ankle1Right = FourBarsActuator(model)
ankle1Right.data = data
ankle1Right.setSerialJoints(
    model.getJointId(ankle1ParentRight), model.getJointId(ankle1OutputRight)
)
ankle1Right.setMotor(idf_m, idf_b)

OM = data.oMf[idf_m].translation
OA = data.oMf[idf_a].translation
OB_up = data.oMf[idf_bup].translation
OB_down = data.oMf[idf_b].translation
assert np.isclose(0, (data.oMf[idf_m].inverse() * OA)[2])
MA = np.linalg.norm(OM[[0, 2]] - OA[[0, 2]])
# AB = np.linalg.norm(OA[[0,2]]-OB_up[[0,2]])
AB = np.linalg.norm(OA - OB_up)
ankle1Right.setMotorArmLength(MA)
ankle1Right.setBindingArmLength(AB)


def dispankle1RightBar(ankle1Right, viz):
    ankle1Right.computeMidPoints()
    inm_a = np.r_[ankle1Right.a, 0]
    in0_a = data.oMf[ankle1Right.idf_m] * inm_a
    in0_b = data.oMf[ankle1Right.idf_m] * ankle1Right.b
    in0_Rbar = pin.Quaternion.FromTwoVectors(
        np.array([0, 0, ankle1Right.l2]), in0_b - in0_a
    ).matrix()
    oMbar = pin.SE3(in0_Rbar, (in0_a + in0_b) / 2)
    viz.applyConfiguration("ankle1Rightbar", oMbar)

    oMrod = pin.SE3(in0_Rbar, in0_a) * pin.SE3(
        pin.utils.rotate("x", -np.pi / 2), np.array([0, -0.04, 0.04])
    )
    viz.applyConfiguration(f"pinocchio/visuals/{ankle1GeomRight}", oMrod)

    viz.applyConfiguration(
        "ankle1Right_a", pin.SE3(data.oMf[ankle1Right.idf_m].rotation, in0_a)
    )
    viz.applyConfiguration(
        "ankle1Right_b", pin.SE3(data.oMf[ankle1Right.idf_b].rotation, in0_b)
    )


# --- ANKLE 1 LEFT
idf_m = model.getFrameId(ankle1FramesLeft[0])
idf_a = model.getFrameId(ankle1FramesLeft[1])
idf_bup = model.getFrameId(ankle1FramesLeft[2])
idf_b = model.getFrameId(ankle1FramesLeft[3])
ankle1Left = FourBarsActuator(model)
ankle1Left.data = data
ankle1Left.setSerialJoints(
    model.getJointId(ankle1ParentLeft), model.getJointId(ankle1OutputLeft)
)
ankle1Left.setMotor(idf_m, idf_b)

OM = data.oMf[idf_m].translation
OA = data.oMf[idf_a].translation
OB_up = data.oMf[idf_bup].translation
OB_down = data.oMf[idf_b].translation
assert np.isclose(0, (data.oMf[idf_m].inverse() * OA)[2])
MA = np.linalg.norm(OM[[0, 2]] - OA[[0, 2]])
# AB = np.linalg.norm(OA[[0,2]]-OB_up[[0,2]])
AB = np.linalg.norm(OA - OB_up)
ankle1Left.setMotorArmLength(MA)
ankle1Left.setBindingArmLength(AB)


def dispankle1LeftBar(ankle1Left, viz):
    ankle1Left.computeMidPoints()
    inm_a = np.r_[ankle1Left.a, 0]
    in0_a = data.oMf[ankle1Left.idf_m] * inm_a
    in0_b = data.oMf[ankle1Left.idf_m] * ankle1Left.b
    in0_Rbar = pin.Quaternion.FromTwoVectors(
        np.array([0, 0, ankle1Left.l2]), in0_b - in0_a
    ).matrix()
    oMbar = pin.SE3(in0_Rbar, (in0_a + in0_b) / 2)
    viz.applyConfiguration("ankle1Leftbar", oMbar)

    oMrod = pin.SE3(in0_Rbar, in0_a) * pin.SE3(
        pin.utils.rotate("x", -np.pi / 2), np.array([0, -0.04, 0.04])
    )
    viz.applyConfiguration(f"pinocchio/visuals/{ankle1GeomLeft}", oMrod)

    viz.applyConfiguration(
        "ankle1Left_a", pin.SE3(data.oMf[ankle1Left.idf_m].rotation, in0_a)
    )
    viz.applyConfiguration(
        "ankle1Left_b", pin.SE3(data.oMf[ankle1Left.idf_b].rotation, in0_b)
    )


# --- RIGHT ANKLE 2
idf_m = model.getFrameId(ankle2FramesRight[0])
idf_a = model.getFrameId(ankle2FramesRight[1])
idf_bup = model.getFrameId(ankle2FramesRight[2])
idf_b = model.getFrameId(ankle2FramesRight[3])
ankle2Right = FourBarsActuator(model)
ankle2Right.data = data
ankle2Right.setSerialJoints(
    model.getJointId(ankle2ParentRight), model.getJointId(ankle2OutputRight)
)
ankle2Right.setMotor(idf_m, idf_b)

OM = data.oMf[idf_m].translation
OA = data.oMf[idf_a].translation
OB_up = data.oMf[idf_bup].translation
OB_down = data.oMf[idf_b].translation
assert np.isclose(0, (data.oMf[idf_m].inverse() * OA)[2])
MA = np.linalg.norm(OM[[0, 2]] - OA[[0, 2]])
# AB = np.linalg.norm(OA[[0,2]]-OB_up[[0,2]])
AB = np.linalg.norm(OA - OB_up)
ankle2Right.setMotorArmLength(MA)
ankle2Right.setBindingArmLength(AB)


def dispankle2RightBar(ankle2Right, viz):
    ankle2Right.computeMidPoints()
    # Point A and B in world frame and M frame
    inm_a = np.r_[ankle2Right.a, 0]
    in0_a = data.oMf[ankle2Right.idf_m] * inm_a
    in0_b = data.oMf[ankle2Right.idf_m] * ankle2Right.b
    in0_Rbar = pin.Quaternion.FromTwoVectors(
        np.array([0, 0, ankle2Right.l2]), in0_b - in0_a
    ).matrix()
    # Orientation matrix aligning AB with Z
    oMbar = pin.SE3(in0_Rbar, (in0_a + in0_b) / 2)
    viz.applyConfiguration("ankle2Rightbar", oMbar)
    # Placement of the existing mesh
    oMrod = pin.SE3(in0_Rbar, in0_a) * pin.SE3(
        pin.utils.rotate("x", -np.pi / 2), np.array([0.025, -0.06, 0.03])
    )
    viz.applyConfiguration(f"pinocchio/visuals/{ankle2GeomRight}", oMrod)

    viz.applyConfiguration(
        "ankle2Right_a", pin.SE3(data.oMf[ankle2Right.idf_m].rotation, in0_a)
    )
    viz.applyConfiguration(
        "ankle2Right_b", pin.SE3(data.oMf[ankle2Right.idf_b].rotation, in0_b)
    )


# --- LEFT ANKLE 2
idf_m = model.getFrameId(ankle2FramesLeft[0])
idf_a = model.getFrameId(ankle2FramesLeft[1])
idf_bup = model.getFrameId(ankle2FramesLeft[2])
idf_b = model.getFrameId(ankle2FramesLeft[3])
ankle2Left = FourBarsActuator(model)
ankle2Left.data = data
ankle2Left.setSerialJoints(
    model.getJointId(ankle2ParentLeft), model.getJointId(ankle2OutputLeft)
)
ankle2Left.setMotor(idf_m, idf_b)

OM = data.oMf[idf_m].translation
OA = data.oMf[idf_a].translation
OB_up = data.oMf[idf_bup].translation
OB_down = data.oMf[idf_b].translation
assert np.isclose(0, (data.oMf[idf_m].inverse() * OA)[2])
MA = np.linalg.norm(OM[[0, 2]] - OA[[0, 2]])
# AB = np.linalg.norm(OA[[0,2]]-OB_up[[0,2]])
AB = np.linalg.norm(OA - OB_up)
ankle2Left.setMotorArmLength(MA)
ankle2Left.setBindingArmLength(AB)


def dispankle2LeftBar(ankle2Left, viz):
    ankle2Left.computeMidPoints()
    # Point A and B in world frame and M frame
    inm_a = np.r_[ankle2Left.a, 0]
    in0_a = data.oMf[ankle2Left.idf_m] * inm_a
    in0_b = data.oMf[ankle2Left.idf_m] * ankle2Left.b
    in0_Rbar = pin.Quaternion.FromTwoVectors(
        np.array([0, 0, ankle2Left.l2]), in0_b - in0_a
    ).matrix()
    # Orientation matrix aligning AB with Z
    oMbar = pin.SE3(in0_Rbar, (in0_a + in0_b) / 2)
    viz.applyConfiguration("ankle2Leftbar", oMbar)
    # Placement of the existing mesh
    oMrod = pin.SE3(in0_Rbar, in0_a) * pin.SE3(
        pin.utils.rotate("x", -np.pi / 2), np.array([0.025, -0.06, 0.03])
    )
    viz.applyConfiguration(f"pinocchio/visuals/{ankle2GeomLeft}", oMrod)

    viz.applyConfiguration(
        "ankle2Left_a", pin.SE3(data.oMf[ankle2Left.idf_m].rotation, in0_a)
    )
    viz.applyConfiguration(
        "ankle2Left_b", pin.SE3(data.oMf[ankle2Left.idf_b].rotation, in0_b)
    )


idg_kneeRight = robot.visual_model.getGeometryId(kneeGeomARight)
idg_ankle1Right = robot.visual_model.getGeometryId(ankle1GeomARight)
idg_ankle2Right = robot.visual_model.getGeometryId(ankle2GeomARight)

idg_kneeLeft = robot.visual_model.getGeometryId(kneeGeomALeft)
idg_ankle1Left = robot.visual_model.getGeometryId(ankle1GeomALeft)
idg_ankle2Left = robot.visual_model.getGeometryId(ankle2GeomALeft)


def createVisuals(viz):
    # ### VIEWER WITH BARS

    # Create viz object for the kneeRight actuator
    viz.addBox("kneeRightbar", [1e-2, 1e-2, kneeRight.l2], "yellow")
    viz.addFrame("kneeRight_a")
    viz.addFrame("kneeRight_b")

    # Create viz object for the kneeLeft actuator
    viz.addBox("kneeLeftbar", [1e-2, 1e-2, kneeLeft.l2], "yellow")
    viz.addFrame("kneeLeft_a")
    viz.addFrame("kneeLeft_b")

    # Create viz object for the ankle 1 actuator
    viz.addBox("ankle1Rightbar", [1e-2, 1e-2, ankle1Right.l2], "yellow")
    viz.addFrame("ankle1Right_a")
    viz.addFrame("ankle1Right_b")

    # Create viz object for the ankle 1 actuator
    viz.addBox("ankle1Leftbar", [1e-2, 1e-2, ankle1Left.l2], "yellow")
    viz.addFrame("ankle1Left_a")
    viz.addFrame("ankle1Left_b")

    # Create viz object for the ankle 2 actuator
    viz.addBox("ankle2Rightbar", [1e-2, 1e-2, ankle2Right.l2], "yellow")
    viz.addFrame("ankle2Right_a")
    viz.addFrame("ankle2Right_b")

    # Create viz object for the ankle 2 actuator
    viz.addBox("ankle2Leftbar", [1e-2, 1e-2, ankle2Left.l2], "yellow")
    viz.addFrame("ankle2Left_a")
    viz.addFrame("ankle2Left_b")


VISUAL_CREATED = False


def dispWithBars(q, viz, createBarsVisuals=True):
    """
    Extend the viz.display(q) function to also display the linkages of the knee.
    """
    global VISUAL_CREATED

    if createBarsVisuals and not VISUAL_CREATED:
        createVisuals(viz)
        VISUAL_CREATED = True

    viz.display(q)
    kneeRight.actuation(q, np.zeros(1))
    dispkneeRightBar(kneeRight, viz)
    kneeLeft.actuation(q, np.zeros(1))
    dispkneeLeftBar(kneeLeft, viz)
    ankle1Right.actuation(q, np.zeros(1))
    dispankle1RightBar(ankle1Right, viz)
    ankle1Left.actuation(q, np.zeros(1))
    dispankle1LeftBar(ankle1Left, viz)
    ankle2Right.actuation(q, np.zeros(1))
    dispankle2RightBar(ankle2Right, viz)
    ankle2Left.actuation(q, np.zeros(1))
    dispankle2LeftBar(ankle2Left, viz)
    # The kneeRight piece is not centered. Hand-tuned value for the center:
    c = np.r_[-0.1713, 0, 0]
    g = robot.visual_model.geometryObjects[idg_kneeRight]
    R = pin.utils.rotate("y", kneeRight.m[0] + 2.05)
    M = pin.SE3(R, c - R @ c)
    viz.applyConfiguration(
        viz.getViewerNodeName(g, pin.VISUAL), viz.visual_data.oMg[idg_kneeRight] * M
    )
    # The kneeLeft piece is not centered. Hand-tuned value for the center:
    c = np.r_[-0.1713, 0, 0]
    g = robot.visual_model.geometryObjects[idg_kneeLeft]
    # added pi here, maybe not necessary
    R = pin.utils.rotate("y", kneeLeft.m[0] + 2.05 - np.pi)
    M = pin.SE3(R, c - R @ c)
    viz.applyConfiguration(
        viz.getViewerNodeName(g, pin.VISUAL), viz.visual_data.oMg[idg_kneeLeft] * M
    )

    g = robot.visual_model.geometryObjects[idg_ankle1Right]
    R = pin.SE3(pin.utils.rotate("z", ankle1Right.m[0] + 1.55), np.zeros(3))
    viz.applyConfiguration(
        viz.getViewerNodeName(g, pin.VISUAL), viz.visual_data.oMg[idg_ankle1Right] * R
    )

    g = robot.visual_model.geometryObjects[idg_ankle1Left]
    R = pin.SE3(pin.utils.rotate("z", ankle1Left.m[0] + 1.55), np.zeros(3))
    viz.applyConfiguration(
        viz.getViewerNodeName(g, pin.VISUAL), viz.visual_data.oMg[idg_ankle1Left] * R
    )

    g = robot.visual_model.geometryObjects[idg_ankle2Right]
    R = pin.SE3(pin.utils.rotate("z", -ankle2Right.m[0] - 0.6), np.zeros(3))
    viz.applyConfiguration(
        viz.getViewerNodeName(g, pin.VISUAL), viz.visual_data.oMg[idg_ankle2Right] * R
    )

    g = robot.visual_model.geometryObjects[idg_ankle2Left]
    R = pin.SE3(pin.utils.rotate("z", -ankle2Left.m[0] - 0.6), np.zeros(3))
    viz.applyConfiguration(
        viz.getViewerNodeName(g, pin.VISUAL), viz.visual_data.oMg[idg_ankle2Left] * R
    )


# sliders.slidersJoints(robot,viz,dispWithBars)

# TODO: problem at the top of the knee, let see later if it is a problem
# for the actuation model.


# Arbitrary order:
"""
hipRight: 0,1,2
hipLeft: 3,4,5
KneeRight: 6
kneeLeft: 7
ankleRight: 8,9
ankleLeft: 10,11
"""

# Serial joints for both hips


class ActuatorSimple:
    def __init__(self, idx_vs, idx_u, nu):
        self.idx_u = idx_u
        self.nu = nu
        self.idx_vs = idx_vs

    def compute_Ja(self):
        return np.eye(3)

    def actuation(self, q, u):
        return u

    def actuationDiff(self, q, u):
        return np.zeros([3, 3])


class FreeFlyer:
    def __init__(self, idx_vs):
        self.idx_u = None
        self.nu = None
        self.idx_vs = idx_vs

    def compute_Ja(self):
        return np.zeros((12, 6))

    def actuation(self, q, u):
        return np.zeros((1, 6))

    def actuationDiff(self):
        return np.zeros((6, 6))


ff = FreeFlyer(idx_vs=[0, 1, 2, 3, 4, 5])
hipRight = ActuatorSimple(
    idx_vs=[model.idx_vs[2], model.idx_vs[3], model.idx_vs[4]], idx_u=0, nu=3
)
hipLeft = ActuatorSimple(
    idx_vs=[model.idx_vs[8], model.idx_vs[9], model.idx_vs[10]], idx_u=3, nu=3
)


q = robot.q0.copy()


def fixNan(nparray):
    # Replace Nan by 0
    nparray[np.isnan(nparray)] = 0
    return nparray


class FullActuation:
    def __init__(
        self,
        model,
        freeflyer,
        hipRight,
        hipLeft,
        kneeRight,
        kneeLeft,
        ankle1Right,
        ankle2Right,
        ankle1Left,
        ankle2Left,
    ):
        # freeflyer + 12 = 18
        self.nv = model.nv

        self.nu = 12

        self.dtau_du = np.zeros([model.nv, self.nu])
        self.dtau_dq = np.zeros([model.nv, model.nv])

        self.freeflyer = freeflyer
        self.hipRight = hipRight
        self.hipLeft = hipLeft
        self.kneeRight = kneeRight
        self.kneeLeft = kneeLeft
        self.ankle1Right = ankle1Right
        self.ankle2Right = ankle2Right
        self.ankle1Left = ankle1Left
        self.ankle2Left = ankle2Left

        self.kneeRight.idx_u, self.kneeRight.nu = 6, 1
        self.ankle1Right.idx_u, self.ankle1Right.nu = 8, 1
        self.ankle2Right.idx_u, self.ankle2Right.nu = 9, 1

        self.kneeLeft.idx_u, self.kneeLeft.nu = 7, 1
        self.ankle1Left.idx_u, self.ankle1Left.nu = 10, 1
        self.ankle2Left.idx_u, self.ankle2Left.nu = 11, 1

        self.actuators = [
            self.freeflyer,
            self.hipRight,
            self.hipLeft,
            self.kneeRight,
            self.kneeLeft,
            self.ankle1Right,
            self.ankle2Right,
            self.ankle1Left,
            self.ankle2Left,
        ]

        self.uTest = np.r_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    def J_a(self, q, u):
        self.dtau_du = np.zeros([model.nv, self.nu])
        self.dtau_dq = np.zeros([model.nv, model.nv])

        for act in self.actuators:
            iu, nu = act.idx_u, act.nu

            if act.idx_u is None:
                # Freeflyer
                Ts = act.actuationDiff()
                Ta = act.compute_Ja()
                self.dtau_du[act.idx_vs, :] = Ta.T
            else:
                Ts = fixNan(act.actuationDiff(q, u[iu : iu + nu]))
                Ta = fixNan(act.compute_Ja())
                self.dtau_du[act.idx_vs, iu : iu + nu] = Ta.T

            # print(act.idx_vs,iu,nu)

            for is_, iv in enumerate(act.idx_vs):
                # print("ooou", Ts[is_, :])
                self.dtau_dq[iv, act.idx_vs] += Ts[is_, :]

        # for iv in range(model.nv):
        #     isActuated = False
        #     for act in actuators:
        #         if iv in act.idx_vs:
        #             isActuated = True
        #             break
        #     if not isActuated:
        #         dtau_du[iv,iv] = 3

        return self.dtau_dq, self.dtau_du

    def dtau_numdiff(self, q, u):
        def actuation(q, u):
            tau = np.zeros(model.nv)
            for act in self.actuators:
                iu, nu = act.idx_u, act.nu
                if act.idx_u is None:
                    # Freeflyer
                    # from IPython import embed
                    # embed()
                    tau[act.idx_vs] += act.actuation(q, u)[0]
                else:
                    tau[act.idx_vs] += act.actuation(
                        q, u[act.idx_u : act.idx_u + act.nu]
                    )
            return tau

        dtau_dq_nd = nd.Jacobian(lambda q_: actuation(q_, u))(q)
        dtau_du_nd = nd.Jacobian(lambda u_: actuation(q, u_))(u)

        return dtau_dq_nd, dtau_du_nd


battobotAct = FullActuation(
    model,
    ff,
    hipRight,
    hipLeft,
    kneeRight,
    kneeLeft,
    ankle1Right,
    ankle2Right,
    ankle1Left,
    ankle2Left,
)


### # ---- Test against Pinocchio


def JcPin(model, data, F1, F2, q):
    """
    Compute the Jacobian of the constraint C between two frames using Pinocchio.
    """
    F1id = model.getFrameId(F1)
    F2id = model.getFrameId(F2)

    pin.forwardKinematics(model, data, q)
    pin.framesForwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pin.computeJointJacobians(model, data, q)
    oMf1 = data.oMf[F1id]
    oMf2 = data.oMf[F2id]
    f1Mf2 = oMf1.actInv(oMf2)

    # return pin.getConstraintJacobian()
    F1jac = pin.getFrameJacobian(model, data, F1id, pin.ReferenceFrame.LOCAL)
    F2jac = pin.getFrameJacobian(model, data, F2id, pin.ReferenceFrame.LOCAL)

    # [0:3] to keep only the translation part (3D constraint)
    return (F1jac - f1Mf2.toActionMatrix() @ F2jac)[0:3]


def JaPin(JC, i_mh, i_s):
    """
    Compute the Jacobian that maps qm_dot to qs_dot using Pinocchio.
    JC: Jacobian of the constraint C
    i_mh: indices of the motor and hidden joints
    i_s: indices of the serial joint
    Return JA: Jacobian that maps qmh_dot to qs_dot
    """
    return (-np.linalg.pinv(JC)[:, i_mh]) @ JC[:, i_s]


# --- Right knee

# JC = JcPin(model, data, kneeFramesRight[2], kneeFramesRight[3], q)
# i_mh = [model.getJointId(kneeParentRight), model.getJointId(kneeOutputRight)]
# i_s = model.getJointId(kneeOutputRight)
# JA = JaPin(JC, i_mh, i_s)
# print(JA)


if __name__ == "__main__":
    print("This is not a standalone script, run test_actuator.py to test this file.")
