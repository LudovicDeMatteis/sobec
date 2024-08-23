######
# This script loads the battobot model and provides some utilities to freeze some of its joints.
# This is standalone from toolbox_parallel_robots.
######


import pinocchio as pin
import numpy as np

def loadBattobot(withFreeFlyer=False):
    urdf = "robot.urdf"
    path = "examples/walk_without_think/model_robot_virgile/model_3d"
    if not withFreeFlyer:
        robot = pin.RobotWrapper.BuildFromURDF(path + "/" + urdf,path)
    else:
        robot = pin.RobotWrapper.BuildFromURDF(path + "/" + urdf,path,
                                               root_joint=pin.JointModelFreeFlyer())
    robot.q0 = pin.neutral(robot.model)
    return robot


# ### NAMES
# kneeFramesRight = [
#     'right_motor_knee',
#     'right_free_transmission_knee',
#     'closedloop_knee_right_A_frame',
#     'closedloop_knee_right_B_frame',
# ]
# kneeFramesLeft = [
#     'left_motor_knee',
#     'left_free_transmission_knee',
#     'closedloop_knee_left_A_frame',
#     'closedloop_knee_left_B_frame',
# ]
# kneeParentRight,kneeOutputRight = 'right_hip_y','right_free_knee'
# kneeParentLeft = 'left_hip_y'
# kneeOutputLeft = 'left_free_knee'
# ankle1FramesRight = [
#     'right_motor_ankle_1',
#     'right_spherical_ankle_1',
#     'closedloop_ankle_right_1A_frame',
#     'closedloop_ankle_right_1B_frame',
# ]
# ankle1FramesLeft = [
#     'left_motor_ankle_1',
#     'left_spherical_ankle_1',
#     'closedloop_ankle_left_1A_frame',
#     'closedloop_ankle_left_1B_frame',
# ]
# ankle1ParentRight,ankle1OutputRight = 'right_free_knee','right_ankle_x'
# ankle1ParentLeft = 'left_free_knee'
# ankle1OutputLeft = 'left_free_ankle_x'
# ankle1ParentLeft = 'left_free_knee'
# ankle1GeomRight = 'part_2_2_0'
# ankle1GeomLeft = 'part_2_0' # <----- ??? not sure
# ankle2FramesRight = [
#     'right_motor_ankle_2',
#     'right_spherical_ankle_2',
#     'closedloop_ankle_right_2A_frame',
#     'closedloop_ankle_right_2B_frame',
# ]
# ankle2FramesLeft = [
#     'left_motor_ankle_2',
#     'left_spherical_ankle_2',
#     'closedloop_ankle_left_2A_frame',
#     'closedloop_ankle_left_2B_frame',

# ]
# ankle2ParentRight,ankle2OutputRight = 'right_free_knee','right_ankle_x'
# ankle2ParentLeft = 'left_free_knee'
# ankle2OutputLeft = 'left_free_ankle_x'
# ankle2ParentLeft = 'left_free_knee'
# ankle2GeomRight = 'small_axis_v2_2_0'
# ankle2GeomLeft = 'small_axis_v2_0' # <----- ??? not sure

# # Rotating attachement
# ankle2GeomARight = 'motor_attachment_4_0' # <------- why _0 ?
# ankle1GeomARight = 'motor_attachment_3_0'
# kneeGeomARight = 'tigh_exo_transmission_2_0'

# ankle2GeomALeft = 'motor_attachment_0'
# ankle1GeomALeft = 'motor_attachment_2_0'
# kneeGeomALeft = 'tigh_exo_transmission_0'


### - TRIAL TO UPDATE THE NAMES TO NEW MODEL

kneeFramesRight = [
    'motor_knee_right',
    'transmission_knee_right',
    'closedloop_knee_right_A_frame',
    'closedloop_knee_right_B_frame',
]
kneeFramesLeft = [
    'motor_knee_left',
    'transmission_knee_left',
    'closedloop_knee_left_A_frame',
    'closedloop_knee_left_B_frame',
]
kneeParentRight,kneeOutputRight = 'hipy_right','transmission_knee_right'
kneeParentLeft = 'hipy_left'
kneeOutputLeft = 'transmission_knee_left'
ankle1FramesRight = [
    'motor_ankle1_right',
    'right_sperical_ankle_1',
    'closedloop_ankle_right_1A_frame',
    'closedloop_ankle_right_1B_frame',
]
ankle1FramesLeft = [
    'motor_ankle1_left',
    'left_spherical_ankle_1',
    'closedloop_ankle_left_1A_frame',
    'closedloop_ankle_left_1B_frame',
]
ankle1ParentRight,ankle1OutputRight = 'transmission_knee_right','anklex_right'
ankle1ParentLeft = 'transmission_knee_left'
ankle1OutputLeft = 'anklex_left'
ankle1ParentLeft = 'transmission_knee_left'
ankle1GeomRight = 'long_axis_assembly'
ankle1GeomLeft = 'long_axis_assembly_2' 
# Warning: ĥ here due to typing error in the model
ankle2FramesRight = [
    'motor_ankle2_right',
    'right_sĥerical_ankle_2',
    'closedloop_ankle_right_2A_frame',
    'closedloop_ankle_right_2B_frame',
]
ankle2FramesLeft = [
    'motor_ankle2_left',
    'left_spherical_ankle_2',
    'closedloop_ankle_left_2A_frame',
    'closedloop_ankle_left_2B_frame',

]
# Warning: why anklex here ?
ankle2ParentRight,ankle2OutputRight = 'transmission_knee_right','anklex_right'
ankle2ParentLeft = 'transmission_knee_left'
ankle2OutputLeft = 'anklex_left'
ankle2ParentLeft = 'transmission_knee_left'
ankle2GeomRight = 'small_axis_assembly'
ankle2GeomLeft = 'small_axis_assembly_2' # <----- ??? not sure

# Rotating attachement
ankle2GeomARight = 'ankle_actuator_assembly_4' # <------- why _0 ?
ankle1GeomARight = 'ankle_actuator_assembly_3'
kneeGeomARight = 'tigh_assembly'

ankle2GeomALeft = 'ankle_actuator_assembly'
ankle1GeomALeft = 'ankle_actuator_assembly_2'
kneeGeomALeft = 'tigh_assembly_2'



###




# ### CORRECT
# ### CORRECT
# ### CORRECT
def alignModel(model,qref):
    '''
    Add some new frames to the Batto model so that they match the expectations
    of the four-bar linkage model:
    - for the planar knee transmission: shift the motor frame on the Y-axis to align it with the binding
    - for each of the 2 ankle transmissions: shift the motor on the Y-axis to align it with the tip of the motor arm
    (point A of our schemas).
    Side effect: modify some of the names of the key frames.
    Warning: remember to rebuild a data afterward, as this function creates new frames in the model.
    Warning: only works on the right leg, TODO on the left
    '''
    data = model.createData()
    pin.framesForwardKinematics(model,data,qref)

    # Bring right knee motor to align it with the binding
    idf_m=model.getFrameId(kneeFramesRight[0])
    idf_b=model.getFrameId(kneeFramesRight[3])
    parent = model.getJointId(kneeParentRight)
    inparent_Mmotor = model.frames[idf_m].placement.copy()
    assert(data.oMi[parent].rotation[1,2]==-1)
    assert(data.oMi[parent+1].rotation[1,2]==+1)
    y_motor = model.jointPlacements[parent+1].translation[2] - model.frames[idf_b].placement.translation[2]
    inparent_Mmotor.translation[2] = y_motor
    model.addFrame(pin.Frame('right_motor_knee_aligned',
                             parent,parent,inparent_Mmotor,pin.OP_FRAME))
    kneeFramesRight[0] = model.frames[-1].name

    # Bring left knee motor to align it with the binding
    idf_m2=model.getFrameId(kneeFramesLeft[0])
    idf_b2=model.getFrameId(kneeFramesLeft[3])
    parent2 = model.getJointId(kneeParentLeft)
    inparent_Mmotor2 = model.frames[idf_m2].placement.copy()
    # TODO
    #assert(data.oMi[parent2].rotation[1,2]==-1)
    #assert(data.oMi[parent2+1].rotation[1,2]==+1)
    y_motor2 = model.jointPlacements[parent2+1].translation[2] - model.frames[idf_b2].placement.translation[2]
    inparent_Mmotor2.translation[2] = y_motor2
    model.addFrame(pin.Frame('left_motor_knee_aligned',
                             parent2,parent2,inparent_Mmotor2,pin.OP_FRAME))
    kneeFramesLeft[0] = model.frames[-1].name


    # Bring right ankle1 motor to align it with attachement on the upper (motor) arm
    idf_m=model.getFrameId(ankle1FramesRight[0])
    idf_a=model.getFrameId(ankle1FramesRight[1])
    parent = model.getJointId(ankle1ParentRight)
    inparent_Mmotor = model.frames[idf_m].placement.copy()
    assert(data.oMi[parent].rotation[1,2]==+1)
    y_motor = model.frames[idf_a].placement.translation[2]
    inparent_Mmotor.translation[2] = y_motor
    inparent_Mmotor.rotation = inparent_Mmotor.rotation@pin.utils.rotate('x',np.pi)
    model.addFrame(pin.Frame(f'{ankle1FramesRight[0]}_aligned',
                             parent,parent,inparent_Mmotor,pin.OP_FRAME))
    ankle1FramesRight[0] = model.frames[-1].name

    # Bring left ankle1 motor to align it with attachement on the upper (motor) arm
    idf_m2=model.getFrameId(ankle1FramesLeft[0])
    idf_a2=model.getFrameId(ankle1FramesLeft[1])
    parent2 = model.getJointId(ankle1ParentLeft)
    inparent_Mmotor2 = model.frames[idf_m2].placement.copy()
    #assert(data.oMi[parent2].rotation[1,2]==+1)
    y_motor2 = model.frames[idf_a2].placement.translation[2]
    inparent_Mmotor2.translation[2] = y_motor2
    inparent_Mmotor2.rotation = inparent_Mmotor2.rotation@pin.utils.rotate('x',np.pi)
    model.addFrame(pin.Frame(f'{ankle1FramesLeft[0]}_aligned',
                             parent2,parent2,inparent_Mmotor2,pin.OP_FRAME))
    ankle1FramesLeft[0] = model.frames[-1].name


    # Bring right ankle2 motor to align it with attachement on the upper (motor) arm
    idf_m=model.getFrameId(ankle2FramesRight[0])
    idf_a=model.getFrameId(ankle2FramesRight[1])
    parent = model.getJointId(ankle2ParentRight)
    inparent_Mmotor = model.frames[idf_m].placement.copy()
    assert(data.oMi[parent].rotation[1,2]==+1)
    y_motor = model.frames[idf_a].placement.translation[2]
    inparent_Mmotor.translation[2] = y_motor
    model.addFrame(pin.Frame(f'{ankle2FramesRight[0]}_aligned',
                             parent,parent,inparent_Mmotor,pin.OP_FRAME))
    ankle2FramesRight[0] = model.frames[-1].name

    # Bring left ankle2 motor to align it with attachement on the upper (motor) arm
    idf_m2=model.getFrameId(ankle2FramesLeft[0])
    idf_a2=model.getFrameId(ankle2FramesLeft[1])
    parent2 = model.getJointId(ankle2ParentLeft)
    inparent_Mmotor2 = model.frames[idf_m2].placement.copy()
    #assert(data.oMi[parent2].rotation[1,2]==+1)
    y_motor2 = model.frames[idf_a2].placement.translation[2]
    inparent_Mmotor2.translation[2] = y_motor2
    model.addFrame(pin.Frame(f'{ankle2FramesLeft[0]}_aligned',
                             parent2,parent2,inparent_Mmotor2,pin.OP_FRAME))
    ankle2FramesLeft[0] = model.frames[-1].name


# ### FREEZE 
# ### FREEZE 
# ### FREEZE 
# Some functions to freeze a part of the robot

def freezeRobot(robot,namesToLock):
    idsToLock = [i for (i, n) in enumerate(robot.model.names) if n in namesToLock ]
    model,(vmodel,gmodel) = \
        pin.buildReducedModel(robot.model,
                              [robot.visual_model,robot.collision_model],
                              idsToLock,robot.q0)
    return pin.RobotWrapper(model,gmodel,vmodel)
    
def freezeSide(robot,key="left"):
    toLock = [ n for n in robot.model.names if key in n ]
    return freezeRobot(robot,toLock)
    
def freezeActuation(robot):
    keys = [
        "motor_ankle_2",
        "spherical_ankle_2",
        "spherical_ankle_1",
        "motor_ankle_1",
        "motor_knee",
        "free_transmission_knee",
    ]
    toLock = [ n for n in robot.model.names if any([ k in n for k in keys ]) ]
    return freezeRobot(robot,toLock)
    

def revoluteToSpherical(robot,joints):
    '''
    Replace the revolute joints in the list by spherical joints.
    '''
    for j in joints:
        idj = robot.model.getJointId(j)
        jmodel = robot.model.joints[idj]
        if jmodel.shortname == 'R':
            jmodel.shortname = 'S'
            jmodel.nq = 4
            jmodel.idx_q = robot.model.nq
            robot.model.nq += 4
            robot.model.nv += 3

if __name__ == "__main__":
    from meshcat_viewer_wrapper.visualizer import MeshcatVisualizer
    robot = loadBattobot()
    robot = freezeSide(robot,'left')
    robot = freezeActuation(robot)

    model = robot.model
    alignModel(model)
    data = model.createData()

    viz = MeshcatVisualizer(robot)
    viz.display(robot.q0)

