import pinocchio as pin
import mujoco
import mujoco.viewer
import numpy as np
import time
from sobec.walk_without_think.battobot import loadBattobot

m = mujoco.MjModel.from_xml_path(
    "/home/vlutz/Documents/sobec_install_sobec4_env/sobec_fork_ludovic/sobec/examples/walk_without_think/model_robot_virgile/model_3d/robot.xml"
)
d = mujoco.MjData(m)

robot = loadBattobot(withFreeFlyer=True)


ACT = [
"left_hip_z",
"left_hip_x",
"left_hip_y",
"left_free_knee",
"left_free_ankle_y",
"left_free_ankle_x",
"right_hip_z",
"right_hip_x",
"right_hip_y",
"right_free_knee",
"right_ankle_y",
"right_ankle_x",
]


def pinToMujocoJointOrder(
    q, model, mujoco_model, mjFloatingBaseName="floating_base_joint"
):
    q_mujoco = np.zeros(len(q))

    pinJointNames = [name for name in model.names]
    pinJointIds = [model.getJointId(name) for name in pinJointNames]
    pinJointIdx_q = [model.joints[i].idx_q for i in pinJointIds]
    pinJoints_nq = [model.joints[i].nq for i in pinJointIds]
    # remove "universe" joint
    pinJointNames.pop(0)
    pinJointIds.pop(0)
    pinJointIdx_q.pop(0)
    pinJoints_nq.pop(0)
    # -------------
    mjJointNames = [mujoco_model.joint(j).name for j in range(mujoco_model.njnt)]
    mjJointIdx_q = np.zeros(len(pinJointIds))
    mjJointIds = np.zeros(len(pinJointIds))
    mjJoints_nq = np.zeros(len(pinJointIds))


    assert len(pinJointNames) == len(mjJointNames)
    assert len(pinJointIds) == len(mjJointIds)

    # This list has to be in bijection with pinJointIds
    # Joint i of pin has index mjJointIds[i] for MuJoCo

    # Check if MJ API allow a better algo
    for i, (id_j, name) in enumerate(zip(pinJointIds, pinJointNames)):
        for j in range(mujoco_model.njnt):
            mj_name = mujoco_model.joint(j).name
            if name == mj_name or (mj_name == mjFloatingBaseName and id_j == 1):
                mjJointIds[i] = j + 1

    print(pinJointNames)
    print(pinJointIds)
    print(pinJointIdx_q)
    print(pinJoints_nq)
    print(mjJointNames)
    print(mjJointIds)

    # from IPython import embed
    # embed()

    for i, mj_id in enumerate(mjJointIds):
        mjJointIdx_q[i] = pinJointIdx_q[int(mj_id - 1)]
        mjJoints_nq[i] = pinJoints_nq[int(mj_id - 1)]

    mjJointIds = mjJointIds.astype(int)
    mjJointIdx_q = mjJointIdx_q.astype(int)
    mjJoints_nq = mjJoints_nq.astype(int)

    print(mjJointIdx_q)
    print(mjJoints_nq)

    for i, (id_j, id_xq, nq, name) in enumerate(
        zip(mjJointIds, mjJointIdx_q, mjJoints_nq, mjJointNames)
    ):
        # from IPython import embed
        # embed()
        try:
            q_mujoco[i : i + nq] = q[id_xq : id_xq + nq]
        except:
            print("Exception")
            print(q_mujoco[i : i + nq])
            print(q[id_xq : id_xq + nq])
            print("i ", i)
            print("nq", nq)
            print("id_xq", id_xq)

    # MuJoCo and Pinocchio uses different quaternion convention
    # permute vectorial part of the quaternion.
    q_mujoco[3] = q[6]
    q_mujoco[4:7] = q[3:6]


    mjActuatorFilter = np.zeros(model.nq)
    for i, name in enumerate(mjJointNames):
        if name in ACT:
            print(name)
            j = mjJointIdx_q[i]
            mjActuatorFilter[j:j+mjJoints_nq[i]] = 1


    #print(q_mujoco)

    return (q_mujoco, mjJointIds, mjJointIdx_q, mjJoints_nq, mjActuatorFilter, mjJointNames)


q0mj, mjJointIds, mjJointIdx_q, mjJoints_nq, mjActuatorFilter, mjJointNames = pinToMujocoJointOrder(
    robot.q0, robot.model, m
)


def turnToStr(q):
    """
    Convert configuration to a string that can be used as a mujoco keyframe.
    """
    outStr = ""
    for q_i in q:
        if q_i < 1e-8:
            outStr += "0.0"
        else:
            outStr += str(q_i)

        outStr += " "

    return outStr


def id_q_is_joint(i, mjJointIdx_q, mjJoints_nq, mjJointNames, jointName):
    # Check if i is in the range of joint jointName
    id_j = mjJointNames.index(jointName)
    if i >= mjJointIdx_q[id_j] and i < mjJointIdx_q[id_j] + mjJoints_nq[id_j]:
        return True
    else:
        return False

Kp = 90*np.ones(12)
Kd = 3*np.ones(12)

xs = np.load(
    "/home/vlutz/Documents/sobec_install_sobec4_env/sobec_fork_ludovic/sobec/examples/walk_without_think/xs.npy"
)
us = np.load(
    "/home/vlutz/Documents/sobec_install_sobec4_env/sobec_fork_ludovic/sobec/examples/walk_without_think/us.npy"
)


with mujoco.viewer.launch_passive(m, d) as viewer:
    # close the viewer after timeout
    start = time.time()
    while viewer.is_running() and time.time() - start < 120:
        step_start = time.time()

        j = 0
        for i, q_i in enumerate(q0mj):

            q = d.qpos[i] 
            
            if i < len(d.qvel) and i >= 0:
                # If out of bounds, it means that the joint is not actuated
                q_dot = d.qvel[i-1]  

            tau_ff = 0 


            if mjActuatorFilter[i] == 1:
                
                ## DEBUG
                # left_free_ankle joints seem to be the problem
                # Need to update the model to the last version.
                # print(j)
                # if "left_free_ankle" in ACT[j]:
                #     Kp[j] = 0.0
                #     Kd[j] = 0.0
                    #print("i ", i, "j ", j, "q ", q, "q0mj ", q0mj[i], "u ", u, "q_dot ", q_dot)

                u = tau_ff + Kp[j] * (q0mj[i]-q) + Kd[j] * (0-q_dot)

                # This suppose that the actuators and joints are in the same order
                d.ctrl[j] = u


                # Check if its true by checking if i is in the range of joint jointName
                assert id_q_is_joint(i, mjJointIdx_q, mjJoints_nq, mjJointNames, ACT[j])
                
                j += 1
                
        mujoco.mj_step(m, d)

        viewer.sync()

        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
