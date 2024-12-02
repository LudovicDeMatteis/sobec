"""
Instanciate the generic actuation models for the specific case of the battobot robot
"""

from .actuation_model import *

# # ### NAMES
kneeFramesRight = [
    "motor_knee_right",
    "transmission_knee_right",
    "closedloop_knee_right_A_frame",
    "knee_right",
]
kneeFramesLeft = [
    "motor_knee_left",
    "transmission_knee_left",
    "closedloop_knee_left_A_frame",
    "knee_left",
]
kneeParentRight, kneeOutputRight = "hipy_right", "knee_right"
kneeParentLeft, kneeOutputLeft = "hipy_left", "knee_left"

ankle1FramesRight = [
    "motor_ankle1_right",
    "right_spherical_ankle_1",
    "closedloop_ankle_right_1B_frame",
    "ankley_right",
]
ankle1FramesLeft = [
    "motor_ankle1_left",
    "left_spherical_ankle_1",
    "closedloop_ankle_left_1B_frame",
    "ankley_left",
]
ankle1ParentRight, ankle1OutputRight = "knee_right", "anklex_right"
ankle1ParentLeft = "knee_left"
ankle1OutputLeft = "anklex_left"
ankle1GeomRight = "long_axis_assembly"
ankle1GeomLeft = "long_axis_assembly_2"

ankle2FramesRight = [
    "motor_ankle2_right",
    "right_spherical_ankle_2",
    "closedloop_ankle_right_2B_frame",
    "ankley_right",
]
ankle2FramesLeft = [
    "motor_ankle2_left",
    "left_spherical_ankle_2",
    "closedloop_ankle_left_2B_frame",
    "ankley_left",
]

ankle2ParentRight, ankle2OutputRight = "knee_right", "anklex_right"
ankle2ParentLeft = "knee_left"
ankle2OutputLeft = "anklex_left"
ankle2GeomRight = "small_axis_assembly"
ankle2GeomLeft = "small_axis_assembly_2"  # <----- ??? not sure

# Rotating attachement
ankle2GeomARight = "ankle_actuator_assembly_1"  # <------- why _0 ?
ankle1GeomARight = "ankle_actuator_assembly_2"
kneeGeomARight = "tigh_assembly_0"

ankle2GeomALeft = "ankle_actuator_assembly_2_3"
ankle1GeomALeft = "ankle_actuator_assembly_2_4"
kneeGeomALeft = "tigh_assembly_2"


# Orientation of the four bar linkage
# can be forward or backward

kneeRightOrientation = "forward"
kneeLeftOrientation = "forward"
ankle1LeftOrientation = "backward"
ankle1RightOrientation = "backward"
ankle2LeftOrientation = "backward"
ankle2RightOrientation = "backward"

def battobot_actuation_factory(robot=None):

    if robot is None:
        # In the context of sobec, cannot be used in this case
        from battobot_loader import loadBattobot
        robot = loadBattobot(withFreeFlyer=True)
        alignModel(robot.model, robot.q0)
        robot = freezeActuation(robot)
    
    model = robot.model
    
    data = model.createData()
    #robot.rebuildData()

    robot.data = data
    pin.framesForwardKinematics(model, robot.data, pin.neutral(model))
    

    # Arbitrary order:
    """
    hipRight: 0,1,2
    hipLeft: 3,4,5
    KneeRight: 6
    kneeLeft: 7
    ankleRight: 8,9
    ankleLeft: 10,11
    """

    # ### ACTUATION MODEL FOR THE FREEFLYER (UNACTUATED)
    ff = FreeFlyer(idx_vs=[0, 1, 2, 3, 4, 5])

    # ### ACTUATION MODELS FOR THE HIPS (DIRECT TRANSMISSION)
    hipRight = ActuatorSimple(
        idx_vs=[model.idx_vs[2], model.idx_vs[3], model.idx_vs[4]], idx_u=0, nu=3
    )
    hipLeft = ActuatorSimple(
        idx_vs=[model.idx_vs[8], model.idx_vs[9], model.idx_vs[10]], idx_u=3, nu=3
    )

    # ### ACTUATION MODELS FOR THE FOUR BARS MECHANISMS
    # Create the actuation models for right and left knee, ankle 1 and 2
    kneeRight = FourBarsActuator(model)
    kneeRight.data = data
    kneeRight.set_type("knee")
    kneeRight.set_rotation_direction("forward")
    kneeRight.setFrames(kneeFramesRight)
    kneeRight.setName("kneeRight")

    kneeLeft = FourBarsActuator(model)
    kneeLeft.data = data
    kneeLeft.set_type("knee")
    kneeLeft.set_rotation_direction("forward")
    kneeLeft.setFrames(kneeFramesLeft)
    kneeLeft.setName("kneeLeft")

    ankle1Right = FourBarsActuator(model)
    ankle1Right.data = data
    ankle1Right.set_type("ankle")
    ankle1Right.set_rotation_direction("backward")
    ankle1Right.setFrames(ankle1FramesRight)
    ankle1Right.setName("ankle1Right")

    ankle1Left = FourBarsActuator(model)
    ankle1Left.data = data
    ankle1Left.set_type("ankle")
    ankle1Left.set_rotation_direction("backward")
    ankle1Left.setFrames(ankle1FramesLeft)
    ankle1Left.setName("ankle1Left")

    ankle2Right = FourBarsActuator(model)
    ankle2Right.data = data
    ankle2Right.set_type("ankle")
    ankle2Right.set_rotation_direction("backward")
    ankle2Right.setFrames(ankle2FramesRight)
    ankle2Right.setName("ankle2Right")

    ankle2Left = FourBarsActuator(model)
    ankle2Left.data = data
    ankle2Left.set_type("ankle")
    ankle2Left.set_rotation_direction("backward")
    ankle2Left.setFrames(ankle2FramesLeft)
    ankle2Left.setName("ankle2Left")

    fourBarsFrames = [
        kneeFramesRight,
        kneeFramesLeft,
        ankle1FramesRight,
        ankle1FramesLeft,
        ankle2FramesRight,
        ankle2FramesLeft,
    ]
    fourBarsParents = [
        kneeParentRight,
        kneeParentLeft,
        ankle1ParentRight,
        ankle1ParentLeft,
        ankle2ParentRight,
        ankle2ParentLeft,
    ]
    fourBarsOutputs = [
        kneeOutputRight,
        kneeOutputLeft,
        ankle1OutputRight,
        ankle1OutputLeft,
        ankle2OutputRight,
        ankle2OutputLeft,
    ]
    fourBarsActuators = [
        kneeRight,
        kneeLeft,
        ankle1Right,
        ankle1Left,
        ankle2Right,
        ankle2Left,
    ]

    # Set actuators lengths and motor positions
    for fbFrames, fbParent, fbOutput, fbActuator in zip(
        fourBarsFrames, fourBarsParents, fourBarsOutputs, fourBarsActuators
    ):
        idf_M = model.getFrameId(fbFrames[0])
        idf_A = model.getFrameId(fbFrames[1])
        idf_B = model.getFrameId(fbFrames[2])
        idf_S = model.getFrameId(fbFrames[3])

        fbActuator.idf_M = idf_M
        fbActuator.idf_A = idf_A
        fbActuator.idf_B = idf_B
        fbActuator.idf_S = idf_S

        fbActuator.setSerialJoints(model.getJointId(fbParent), model.getJointId(fbOutput))
        fbActuator.setMotor(idf_M, idf_B)
        OM = data.oMf[idf_M].translation
        OA = data.oMf[idf_A].translation
        OS = data.oMf[idf_S].translation
        OB = data.oMf[idf_B].translation
        MA = np.linalg.norm(OM - OA)
        AB = np.linalg.norm(OA - OB)
        BS = np.linalg.norm(OB - OS)
        MS = np.linalg.norm(OM - OS)
        fbActuator.set_l1(MA)
        fbActuator.set_l2(AB)
        fbActuator.set_l2_bar(AB)
        fbActuator.set_l3(BS)
        fbActuator.set_l4(MS)


    # ### ASSEMBLE ACTUATION MODELS INTO FULL ACTUATION MODEL

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

    return robot, battobotAct
