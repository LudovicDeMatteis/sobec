"""
# Acuation models for four bars actuators and simple actuators (direct transmission and freeflyer)
# Classes can be instanciated to build actuation model for a given robot
# List of classes:
# - FourBarsActuator: Actuation model for a four bars actuator
# - ActuatorSimple: Actuation model for a simple actuator (direct transmission)
# - FreeFlyer: Actuation model for the freeflyer (unactuated)
# - FullActuation: Actuation model for the full actuation of the robot
"""

import matplotlib.pyplot as plt
import numdifftools as nd  # pip install numdifftools
import numpy as np
import pinocchio as pin
from numpy.linalg import norm
from numpy.testing import assert_almost_equal
from pinocchio.utils import rotate


sumsqr = lambda v: sum(v**2)
sqrt = np.sqrt
arccos, arcsin, arctan, arctan2 = np.arccos, np.arcsin, np.arctan, np.arctan2
A_ = np.array
cos, sin = np.cos, np.sin
array = np.array


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
        self.rotationDirection = "cw"
        self.nonSanityCount = 0

    def set_l1(self, l1):
        """
        Length of the bar attached to the motor (i.e. MA on our drawings)
        """
        self.l1 = l1

    def set_l2(self, l2):
        """
        Length of the bar attached to the (free) bindings (i.e. AB on our drawings)
        """
        self.l2 = l2

    def set_l2_bar(self, l2_b):
        """
        Length of the projected l2. In case of real planar 4 bar
        (ie knee) l2 = l2_bar. For the ankle, this is not the case
        l2_bar need to be used for visualization
        """
        self.l2b = l2_b

    def set_l3(self, l3):
        """
        Length of BS on our drawings
        """
        self.l3 = l3

    def set_l4(self, l4):
        """
        Length of MS on our drawings
        """
        self.l4 = l4

    def set_type(self, actuatorType):
        """
        Type of four bar : "knee" or "ankle".
        Knee is static, while ankle varies with configuration
        """
        self.actuatorType = actuatorType

    def set_rotation_direction(self, direction):
        """
        Set rotation direction of the motor. Can be "forward" or "backward"
        """
        self.rotationDirection = direction

    def setSerialJoints(self, parentId, endId):
        assert parentId < len(self.model.joints)
        assert endId < len(self.model.joints)
        self.idj_parent = parentId
        self.idj_end = j = endId
        self.serialJoints = []
        # Debug
        t = 0
        while True:
            t += 1
            assert self.model.nvs[j] == 1
            self.serialJoints.insert(0, j)
            j = self.model.parents[j]
            if j == self.idj_parent:
                break
            if t > 10:
                raise ValueError(
                    "Too many iterations, parentId or endId are likely wrong"
                )

        self.idx_qs = [self.model.idx_qs[i] for i in self.serialJoints]
        self.idx_vs = [self.model.idx_vs[i] for i in self.serialJoints]

    def setMotor(self, motorFrameId, bindingFrameId):
        assert motorFrameId < len(self.model.frames)
        assert bindingFrameId < len(self.model.frames)
        self.idf_M = motorFrameId
        self.idf_B = bindingFrameId

    def setFrames(self, framesList):
        """
        Set the frames for the actuator
        framesList: list of frame names
        """
        for i, frameName in enumerate(framesList):
            assert frameName in [frame.name for frame in self.model.frames]

        self.frames = framesList

    def setName(self, name):
        """
        set the name of the actuator.
        Used to set names for visuals in the visualizer
        name: string
        """
        self.name = name

    def fk(self, s):
        """
        Compute the world position of the linkages, and the corresponding derivative wrt s
        That should be evaluated in motor frame ... TODO
        For convenience, return the pairs (b,Bs)
        """
        pin.forwardKinematics(self.model, self.data, s)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data)

        self.inm_M0 = self.data.oMf[self.idf_M].inverse()

        self.b = (self.inm_M0 * self.data.oMf[self.idf_B]).translation

        self.Bs = self.inm_M0.rotation @ self.data.J[:3, self.idx_vs]
        self.Bs -= (
            pin.skew(self.b - self.inm_M0.translation)
            @ self.inm_M0.rotation
            @ self.data.J[3:, self.idx_vs]
        )

        if self.withSanityCheck:
            self.kineCheck(s)  # Sanity check

        return self.b, self.Bs

    def actuation(self, s, u, updateLengths=None, case="normal"):
        # Compute the angle m
        b, Bs = self.fk(s)
        self.b = b

        # Only for visual purposes
        if updateLengths is not None:
            updateLengths()

        l1, l2 = self.l1, self.l2

        l__2 = self.l__2 = sumsqr(b[:2])
        l = self.l = np.sqrt(l__2)
        l2b__2 = self.l2b__2 = l2**2 - b[2] ** 2
        self.l2b = sqrt(l2b__2)

        rc = self.rc = (l__2 + l1**2 - l2b__2) / (2 * l * l1)
        rs = self.rs = sqrt(1 - rc**2)

        m1 = arctan2(b[1], b[0])
        # arccos here in the formulas we wrote
        m2 = arctan2(rs, rc)
        self.m1 = m1
        self.m2 = m2
        self.m = np.r_[m1 + m2]

        # Sanity check
        checkLogs = [
            "l2b__2 < 0",
            "1-rc**2 < 0",
            "b[0] < 0",
            "b[1] < 0",
        ]
        check1 = l2b__2 < 0
        check2 = 1 - rc**2 < 0
        check3 = b[0] < 0
        check4 = b[1] < 0
        # Current toughts:
        # Numdiff algo (numdifftools) has variable step size, leading to out of bounds q
        # Numdiff code create out of bounds q, which leads to out of bounds b
        # It may be not a problem ? Assertions look good
        # ------
        # By setting the stepsize manually, we can avoid getting out of bounds for q
        # But then the precision of the numdiff is not as good
        # Conclusion for now:
        # ----> Avoid warning in case of numdiff
        if (check1 or check2 or check3 or check4) and case != "numdiff":
            self.nonSanityCount += 1
            print(case + " mode", "q=" + str(s), "Length of q: ", len(s))
            listOfFailures = [check1, check2, check3, check4]
            print(
                self.actuatorType
                + " sanity failure due to check:"
                + str([i + 1 for i, x in enumerate(listOfFailures) if x]),
                [checkLogs[i] for i, x in enumerate(listOfFailures) if x],
            )

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

    def actuationDiff(self, s, u, updateLengths=None, case="normal"):
        # Bs.T @ K.T @ b @ u
        self.actuation(s, u, updateLengths, case)

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
            self.actuation(s[:, 0], np.array([0]), case="numdiff")
            return self.m

        return nd.Jacobian(s_to_m)(s[:, np.newaxis])

    def actuationNumdiff(self, s, u):
        As_ = nd.Jacobian(lambda s: self.actuation(s, u, case="numdiff"))
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
        Dirty hack for debug, do not use if you are not me (NM)
        """
        for attr in dir(self):
            if not attr.startswith("__") and not callable(getattr(self, attr)):
                loc[attr] = getattr(self, attr)

    def dispFrames(self, viz):
        """
        display frames of the four bar
        viz: meshcat visualizer
        """
        for n in self.frames:
            i = self.model.getFrameId(n)
            print("\n\n---")
            print(n, i)
            print(self.data.oMf[i])
            viz.applyConfiguration("/f", self.data.oMf[i])
            input()

    def createVisuals(self, viz):
        # Create geoms for the four bars
        lengths = [self.l1, self.l2b, self.l3, self.l4]
        for j, length in enumerate(lengths):
            currentGeom = self.name + "bar_"
            viz.addBox(f"{currentGeom}{j+1}", [1e-2, 1e-2, length], "yellow")
        viz.addFrame(f"{currentGeom}a")
        viz.addFrame(f"{currentGeom}b")
        self.visual_created = True

    def dispFourBars(self, viz):
        """
        Display the four bars of the actuator in yellow in the meshcat visualizer
        Visuals need to be created before calling this function, with the createVisuals method
        viz: meshcat visualizer
        """
        assert self.visual_created
        idf_M = self.model.getFrameId(self.frames[0])
        # A is not correctly placed since we are working on the reduced model (except in q0)
        idf_A = self.model.getFrameId(self.frames[1])
        idf_B = self.model.getFrameId(self.frames[2])
        idf_S = self.model.getFrameId(self.frames[3])
        # couples of frames that hold the bars
        bars = [
            [idf_M, idf_A],
            [idf_A, idf_B],
            [idf_B, idf_S],
            [idf_S, idf_M],
        ]

        for i, (f1, f2) in enumerate(bars):
            oMbar = pin.SE3(1)

            # for the bars that involve the serial chain, we use the model frames oMf
            # for the other bars (MA) and (AB), we need to compute the virtual pose of A.
            # i.e, the pose of A if we would have the closed chain
            if (f1, f2) == (idf_M, idf_A) or (f1, f2) == (idf_A, idf_B):
                # get the frame that is not A:
                if f1 == idf_A:
                    otherFrame = f2
                else:
                    otherFrame = f1
                # get the pose of M:
                oM_M = self.data.oMf[idf_M].copy()
                oM_S = self.data.oMf[idf_S].copy()

                M_M_S = oM_M.actInv(oM_S)
                M_MS = M_M_S.translation
                M_MS_scaled = M_MS / norm(M_MS) * self.l1

                # using the computed m angle, compute the pose of A from M
                m_angle = self.m
                # A has an easy expression in the motor frame
                # scale SM to l1, rotate around z from m
                # transform SM_scaled to the motor frame
                M_MA = pin.SE3(np.eye(3), rotate("z", float(m_angle)) @ M_MS_scaled)
                oM_A = oM_M.act(M_MA)
                # debug
                viz.addBox("A_" + self.name, [1e-2, 1e-2, 1e-2], "red")
                viz.applyConfiguration("A_" + self.name, oM_A)

                oMbar.translation = (
                    self.data.oMf[otherFrame].translation + oM_A.translation
                ) / 2
                # find the new frame that holds the bar in the right orientation
                # (z is the direction of the bar), its origin is at equal distance of the two frames
                z_newFrame = (
                    self.data.oMf[otherFrame].translation - oM_A.translation
                ) / norm((self.data.oMf[otherFrame].translation - oM_A.translation))
            else:
                oMbar.translation = (
                    self.data.oMf[f1].translation + self.data.oMf[f2].translation
                ) / 2
                # find the new frame that holds the bar in the right orientation
                # (z is the direction of the bar), its origin is at equal distance of the two frames
                z_newFrame = (
                    self.data.oMf[f2].translation - self.data.oMf[f1].translation
                ) / norm(self.data.oMf[f2].translation - self.data.oMf[f1].translation)
            x_newFrame = np.cross([0, 0, 1], z_newFrame)
            x_newFrame = x_newFrame / np.linalg.norm(x_newFrame)
            y_newFrame = np.cross(z_newFrame, x_newFrame)
            y_newFrame = y_newFrame / np.linalg.norm(y_newFrame)
            oMbar.rotation = np.c_[x_newFrame, y_newFrame, z_newFrame]

            barName = f"{self.name}bar_{i+1}"

            viz.applyConfiguration(barName, oMbar)

    def plot_four_bar(self, direction="cw"):
        from matplotlib.patches import Arc

        m1 = self.m1
        m2 = self.m2
        l3 = self.l3
        l4 = self.l4
        l1 = self.l1
        l2 = self.l2
        b = self.b

        # Coordonnées de la barre l4 (horizontale)
        x1, y1 = 0, 0
        x2, y2 = l4, 0

        if direction == "cw":
            m = m1 + m2
        else:
            m = m1 - m2

        # Coordonnées de la barre l3, accroché à gauche de l4 avec un angle m
        x3 = l3 * np.cos(m)
        y3 = l3 * np.sin(m)

        # Tracé de la barre l4
        plt.plot([x1, x2], [y1, y2], "bo-", label=f"l4 = {l4}")

        # Tracé de la barre l3
        plt.plot(
            [x1, x3],
            [y1, y3],
            "ro-",
            label=f"l3 = {l3}, angle = {np.degrees(m):.2f}° / {m:.2f} rad ",
        )

        # coord b=[xB,yB]
        xB = b[0]
        yB = b[1]
        plt.plot([x3, xB], [y3, yB], "go-", label="b, l2")

        plt.plot([x2, xB], [y2, yB], "yo-", label="s, l1")

        # Ajout de l'arc de cercle représentant l'angle m
        angle_deg = np.degrees(m)  # conversion de l'angle en degrés
        arc = Arc(
            (0, 0),
            l3 / 2,
            l3 / 2,
            theta1=0,
            theta2=angle_deg,
            color="purple",
            lw=2,
            label="Angle m " + str(np.degrees(m)) + "deg",
        )
        plt.gca().add_patch(arc)

        arc_m1 = Arc(
            (0, 0),
            l3 / 2.2,
            l3 / 2.2,
            theta1=0,
            theta2=np.degrees(m1),
            color="orange",
            lw=2,
            label="Angle m2 " + str(np.degrees(m1)) + "deg",
        )
        plt.gca().add_patch(arc_m1)
        arc_m2 = Arc(
            (0, 0),
            l3 / 2.4,
            l3 / 2.4,
            theta1=np.degrees(m1),
            theta2=np.degrees(m2) + np.degrees(m1),
            color="green",
            lw=2,
            label="Angle m2 " + str(np.degrees(m2)) + "deg",
        )
        plt.gca().add_patch(arc_m2)

        plt.plot([0, xB], [0, yB], "--", label="l")

        # Textual information
        plt.text(x1, y1, "M", fontsize=12, ha="right")
        plt.text(x2, y2, "S", fontsize=12, ha="left")
        plt.text(x3, y3, "A", fontsize=12, ha="right")
        plt.text(xB, yB, "B", fontsize=12, ha="right")

        # Réglages du graphique
        plt.xlim(-l3, l4 + l3)
        plt.ylim(-l3, l3)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(True)
        plt.legend()
        plt.title(self.name + " 4 bar linkage")

        plt.show()


class ActuatorSimple:
    def __init__(self, idx_vs, idx_u, nu):
        self.idx_u = idx_u
        self.nu = nu
        self.idx_vs = idx_vs
        self.actuatorType = "simple"

    def compute_Ja(self):
        return np.eye(3)

    def compute_Ja_numdiff(self, q):
        # no need to numdiff here, just an alias
        return self.compute_Ja()

    def actuation(self, q, u, case="None"):
        return u

    def actuationDiff(self, q, u, updateActuatorsLengths=None, case=None):
        if updateActuatorsLengths is not None:
            updateActuatorsLengths()
        return np.zeros([3, 3])


class FreeFlyer:
    def __init__(self, idx_vs):
        self.idx_u = None
        self.nu = None
        self.idx_vs = idx_vs
        self.actuatorType = "freeflyer"

    def compute_Ja(self):
        return np.zeros((12, 6))

    def actuation(self, q, u, case="None"):
        return np.zeros((1, 6))

    def actuationDiff(self, updateActuatorsLengths=None, case=None):
        if updateActuatorsLengths is not None:
            updateActuatorsLengths()
        return np.zeros((6, 6))


def fixNan(nparray):
    # no nan for now, we may remove this function later
    # nparray[np.isnan(nparray)] = 0
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

        self.model = model
        self.data = model.createData()

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

        self.visual_created = False
        self.uTest = np.ones(12)

    def updateActuatorsLengths(self):
        """
        Warning: Obsolete. Could be useful to update the visuals only. TODO
        Should be called after frameForwardKinematics
        """
        for act in [
            actuator for actuator in self.actuators if actuator.actuatorType == "ankle"
        ]:
            if isinstance(act, FourBarsActuator):
                OM = act.data.oMf[act.idf_M].translation
                OA = act.data.oMf[act.idf_A].translation
                OS = act.data.oMf[act.idf_S].translation
                OB = act.data.oMf[act.idf_B].translation
                MA = np.linalg.norm(OM - OA)
                AB = np.linalg.norm(OA - OB)
                BS = np.linalg.norm(OB - OS)
                MS = np.linalg.norm(OM - OS)

                # print(act,MA,AB,BS,MS)
                # After thinking again,
                # it seems that this function is pointless
                # if not used only for the visuals.
                # Plus, it breaks the computation
                # Will be updated in the future
                # act.set_l1(MA)
                # act.set_l2(AB)
                # act.set_l3(BS)
                # act.set_l4(MS)

    def J_a(self, q, u, numdiff=False):
        self.dtau_du = np.zeros([self.model.nv, self.nu])
        self.dtau_dq = np.zeros([self.model.nv, self.model.nv])

        for act in self.actuators:
            iu, nu = act.idx_u, act.nu

            if act.idx_u is None:
                # Freeflyer
                if numdiff:
                    Ts = act.actuationDiff(case="numdiff")
                else:
                    Ts = act.actuationDiff(self.updateActuatorsLengths)
                Ta = act.compute_Ja()
                self.dtau_du[act.idx_vs, :] = Ta.T
            else:
                if numdiff:
                    Ts = fixNan(act.actuationDiff(q, u[iu : iu + nu], case="numdiff"))
                else:
                    Ts = fixNan(
                        act.actuationDiff(
                            q, u[iu : iu + nu], self.updateActuatorsLengths
                        )
                    )
                Ta = fixNan(act.compute_Ja())
                self.dtau_du[act.idx_vs, iu : iu + nu] = Ta.T
                print

            # print(act.idx_vs,iu,nu)

            for is_, iv in enumerate(act.idx_vs):
                self.dtau_dq[iv, act.idx_vs] += Ts[is_, :]

        return self.dtau_dq, self.dtau_du

    def J_a_numdiff(self, q, u):
        Ja = np.zeros([self.nv, self.nu])
        for i, act in enumerate(self.actuators):
            iu, nu = act.idx_u, act.nu
            if act.idx_u is None:
                # Freeflyer
                Ja[act.idx_vs, :] = act.compute_Ja().T
            else:
                try:
                    Ja[act.idx_vs, iu : iu + nu] = act.compute_Ja_numdiff(q).T
                except:
                    Ja[act.idx_vs, iu : iu + nu] = act.compute_Ja_numdiff(q).T[
                        act.idx_qs
                    ]
        return Ja

    def dtau_double_numdiff(self, q, u):
        def actuation(q, u):
            du = self.J_a_numdiff(q, u)
            return du @ u

        dtau_dq_nd = nd.Jacobian(lambda q_: actuation(q_, u))(q)
        dtau_du_nd = nd.Jacobian(lambda u_: actuation(q, u_))(u)

        return dtau_dq_nd, dtau_du_nd

    def dtau_numdiff(self, q, u):
        # def actuation(q, u):
        #     tau = np.zeros(model.nv)
        #     for act in self.actuators:
        #         iu, nu = act.idx_u, act.nu
        #         if act.idx_u is None:
        #             # Freeflyer
        #             # from IPython import embed
        #             # embed()
        #             tau[act.idx_vs] += act.actuation(q, u)[0]
        #         else:
        #             act.actuationDiff(q, u[iu : iu + nu])
        #             tau[act.idx_vs] += act.actuation(
        #                 q, u[act.idx_u : act.idx_u + act.nu]
        #             )

        #     return tau

        def actuation(q, u):
            _, du = self.J_a(q, u, numdiff=True)
            return du @ u

        dtau_dq_nd = nd.Jacobian(lambda q_: actuation(q_, u))(q)
        dtau_du_nd = nd.Jacobian(lambda u_: actuation(q, u_))(u)

        return dtau_dq_nd, dtau_du_nd

    def createVisuals(self, viz):
        """
        Create visuals for all the actuators
        """
        for fb in self.actuators:
            if isinstance(fb, FourBarsActuator):
                fb.createVisuals(viz)
        self.visual_created = True

    def dispWithBars(self, q, viz, createBarsVisuals=True, vanilla=False):
        """
        Extend the viz.display(q) function to also display the linkages of the knee.
        """

        if createBarsVisuals and not self.visual_created:
            self.createVisuals(viz)

        viz.display(q)

        for fb in self.actuators:
            fb.actuation(q, np.zeros(1), case="visual")
            if isinstance(fb, FourBarsActuator):
                fb.dispFourBars(viz)

        # dispAllFrames(model,data,viz)


### # ---- Test against Pinocchio
# Exemple routine from old repo that need to be adapted
# Current thoughts:
# I think this is not possible as long as the yellow visuals are not set as a Pinocchio model with two kinematic trees


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
    return (-np.linalg.pinv(JC)) @ JC[:, i_s]


if __name__ == "__main__":
    print("This is not a standalone script, run test_actuator.py to test this file.")
