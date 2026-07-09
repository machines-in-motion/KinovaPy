import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pinocchio as pin


def plotJointTrajectory(xs=None, us=None, dt=0.02, figIndex=1, show=True, figTitle=''):
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # Getting the state and control trajectories
    if xs is not None:
        qsPlotIdx = 211
        vsPlotIdx = 212
        nq = 7
        nx = 14
        X = [0.0] * nx
        for i in range(nx):
            X[i] = [x[i] for x in xs]
    if us is not None:
        usPlotIdx = 111
        nu = us[0].shape[0]
        U = [0.0] * nu
        for i in range(nu):
            U[i] = [u[i] if u.shape[0] != 0 else 0 for u in us]
    if xs is not None and us is not None:
        qsPlotIdx = 311
        vsPlotIdx = 312
        usPlotIdx = 313

    plt.figure(figIndex)
    plt.suptitle(figTitle)

    # Plotting the state trajectories
    times = np.arange(xs.shape[0])*dt
    if xs is not None:
        plt.subplot(qsPlotIdx)
        [plt.plot(times, X[i], label="q" + str(i)) for i in range(nq)]
        plt.legend()
        plt.ylabel('Joint position [rad]')

    if xs is not None:
        plt.subplot(vsPlotIdx)
        [plt.plot(times, X[i+7], label="v" + str(i)) for i in range(nq)]
        plt.legend()
        plt.ylabel('Joint velocity [rad/s]')

    # Plotting the actual controls
    if us is not None:
        plt.subplot(usPlotIdx)
        [plt.plot(times, U[i], label="u" + str(i)) for i in range(nu)]
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel('Joint torque [Nm]')
    if show:
        plt.show()


def plotSolutionVsActual(x_des, u_des, xs, us, dt=0.02, joint_id=0, show=True):
    """
    - x_all (array[N_sim,N_h,nq]): all state's solution history
    - u_all (array[N_sim,N_h,nu]): all control's solution history
    - xs (array[N_sim,nq]): actual state trajectory
    - us (array[N_sim,nu]): actual control trajectory
    - x_id (int): which state's solution history to be plotted (hair plot)
    - u_id (int): which control's solution history to be plotted (hair plot)
    - plot_length (int): how many knots in the hair plot
    """
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # Plot solution history (hair plot)
    u_id = joint_id
    x_id = joint_id
    xi_sol = x_des[:-1,x_id]
    ui_sol = u_des[:-1,u_id]
    xi_s = xs[1:,x_id]
    ui_s = us[1:,u_id]
    times =np.arange(xs.shape[0]-1)*dt
    plt.figure()
    plt.suptitle("Solution versus Actual state/control of joint q%s"%joint_id)
    plt.subplot(2,1,1)
    plt.plot(times, xi_s, 'b-', label='Actual value')
    plt.plot(times, xi_sol, 'r--', label='MPC solution')
    plt.xlabel("Time [s]")
    plt.ylabel('Joint value  $q_{%s}$'%joint_id)
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(times, ui_s, 'b-', label='Actual value')
    plt.plot(times, ui_sol, 'r--', label='MPC solution')
    plt.xlabel("Time [s]")
    plt.ylabel(r'Torque  $\tau_{%s}$'%joint_id)
    plt.legend()

    # Show plots
    if show:
        plt.show()


def plotFrameTrajectory(rmodel, xs, ee_id, goal_pos=None, label='End Effector', show=True):
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    rdata = rmodel.createData()
    ee = np.empty((len(xs), 3))
    for i in range(len(xs)):
        x = xs[i]
        q = x[:rmodel.nq]
        pin.forwardKinematics(rmodel, rdata, q)
        pin.updateFramePlacements(rmodel, rdata)
        if ee_id is not None:
            ee[i] = rdata.oMf[ee_id].translation

    plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.title("End-effector trajectory")
    if goal_pos is not None:
        plt.plot(goal_pos[0], goal_pos[1], marker = '*', markersize = 20, color = 'g')
    plt.plot(ee[:,0], ee[:,1], 'b-', label=label)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()

    plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.title("End-effector trajectory")
    if goal_pos is not None:
        plt.plot(goal_pos[1], goal_pos[2], marker = '*', markersize = 20, color = 'g')
    plt.plot(ee[:,1], ee[:,2], 'b-', label=label)
    plt.xlabel("y [m]")
    plt.ylabel("z [m]")
    plt.legend()

    plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.title("End-effector trajectory")
    if goal_pos is not None:
        plt.plot(goal_pos[0], goal_pos[2], marker = '*', markersize = 20, color = 'g')
    plt.plot(ee[:,0], ee[:,2], 'b-', label=label)
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.legend()

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(ee[:,0], ee[:,1], ee[:,2], label=label)
    ax.scatter(ee[0,0], ee[0,1], ee[0,2], color='green', marker='o', label='Start')
    ax.scatter(ee[-1,0], ee[-1,1], ee[-1,2], color='red', marker='x', label='End')
    if goal_pos is not None:
        ax.scatter(goal_pos[0], goal_pos[1], goal_pos[2], color='g', marker='*', label='Goal')
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title('End-Effector Trajectory')
    ax.legend()

    # Show plots
    if show:
        plt.show()


def plotComputeTime(sol_times, dt, show=True):
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    times = np.arange(sol_times.shape[0])*dt
    plt.figure()
    plt.plot(times, sol_times, label='Solve time')
    plt.ylabel('Compute time [s]')
    plt.xlabel('Time [s]')
    plt.legend()
    # Show plots
    if show:
        plt.show()