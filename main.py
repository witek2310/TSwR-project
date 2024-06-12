import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
from numba import jit
# Create the environment
import time



def testy(theta_num, theta_dot_num, actions_num, masa):
    # Parameters for the pendulum (assuming default values used in Gym's Pendulum-v1)
    m = 1.0  # mass of the pendulum
    l = 1.0  # length of the pendulum
    g = 9.81  # acceleration due to gravity
    dt = 0.05  # time step for integration
    env = gym.make('Pendulum-v1', g=g)

    # Discretization parameters
    num_theta = theta_num
    num_thetadot = theta_dot_num
    num_actions = actions_num

    # Bounds for state variables
    theta_min, theta_max = -np.pi, np.pi
    thetadot_min, thetadot_max = -8, 8
    action_min, action_max = env.action_space.low[0], env.action_space.high[0]

    # Discretize state space
    theta_grid = np.linspace(theta_min, theta_max, num_theta)
    thetadot_grid = np.linspace(thetadot_min, thetadot_max, num_thetadot)
    actions = np.linspace(action_min, action_max, num_actions)

    
    # Parameters

    theta_step = (theta_max - theta_min) / (num_theta - 1)
    thetadot_step = (thetadot_max - thetadot_min) / (num_thetadot - 1)

    # Helper functions
    @jit()
    def get_state_index(state):
        theta, thetadot = state
        theta_idx = int(round((theta - theta_min) / theta_step))
        thetadot_idx = int(round((thetadot - thetadot_min) / thetadot_step))
        if theta_idx < 0:
            theta_idx = 0
        elif theta_idx >= num_theta:
            theta_idx = num_theta - 1
            
        if thetadot_idx < 0:
            thetadot_idx = 0
        elif thetadot_idx >= num_thetadot:
            thetadot_idx = num_thetadot - 1
        return theta_idx, thetadot_idx
    @jit()
    def get_next_state(state, action):
        
        theta, thetadot = state
        u = action

        reward = - ((theta)**2 + 0.1 * (thetadot**2) + 0.001 * (u**2))
        newthdot = thetadot + (3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l**2) * u) * dt

        if newthdot < thetadot_min:
            newthdot = thetadot_min
        elif newthdot > thetadot_max:
            newthdot = thetadot_max
        next_theta = theta + newthdot * dt

        # Ensure the angle stays within the range [-pi, pi]
        next_theta = (next_theta + np.pi) % (2 * np.pi) - np.pi



       
        
        return (next_theta, newthdot), reward

    @jit()
    def value_iteration(V, theta_grid, thetadot_grid, actions, gamma, max_iterations=1000, tol=1e-4):
        policy = np.zeros((num_theta, num_thetadot))
        for i in range(max_iterations):
            # print(i)
            delta = 0
            new_V = np.copy(V)
            for theta_idx, theta in enumerate(theta_grid):
                for thetadot_idx, thetadot in enumerate(thetadot_grid):
                    state = (theta, thetadot)
                    best_value = -10000000.0
                    for action in actions:
                        next_state, reward = get_next_state(state, action)
                        next_theta_idx, next_thetadot_idx = get_state_index(next_state)
                        value = reward + gamma * V[next_theta_idx, next_thetadot_idx]
                        if value > best_value:
                            best_value = value
                            best_action = action
                    new_V[theta_idx, thetadot_idx] = best_value
                    policy[theta_idx, thetadot_idx] = best_action
                    delta = max(delta, abs(new_V[theta_idx, thetadot_idx] - V[theta_idx, thetadot_idx]))
            V = new_V
            if delta < tol:
                break
        return V, policy

    gamma = 0.99
    V = np.zeros((num_theta, num_thetadot))
    start_time = time.time_ns()

    V, policy = value_iteration(V, theta_grid, thetadot_grid, actions, gamma, 4000)
    end_time = time.time_ns()
    eval_time = end_time - start_time
    eval_time = eval_time * 1e-9
    eval_time = round(eval_time, 4)
    fig = plt.figure(figsize=(9, 4))
    ax1, ax2 = fig.subplots(1, 2)
    ax1.set_xlabel("q")
    ax1.set_ylabel("qdot")
    ax1.set_title("Cost-to-Go")
    ax2.set_xlabel("q")
    ax2.set_ylabel("qdot")
    ax2.set_title("Policy")
    ax1.imshow(
        V.transpose(),
        aspect="auto",
        extent=(theta_grid[0], theta_grid[-1], thetadot_grid[0], thetadot_grid[-1])
    )
    ax1.invert_yaxis()
    # Pi = np.reshape(policy.get_output_values(), Q.shape)
    ax2.imshow(
        policy.transpose(),
        aspect="auto",
        extent=(theta_grid[0], theta_grid[-1], thetadot_grid[0], thetadot_grid[-1])
    )
    ax2.invert_yaxis()
    plt.suptitle(f"thetas: {num_theta}, thetas_dot: {num_thetadot}, actions: {num_actions} , t: {eval_time}s", size=14)
    # plt.savefig(f"graphs/pol_{num_theta}_{num_thetadot}_{num_actions}.pdf")
    plt.show()
    # plt.close()
    obs,_ = env.reset()

    num__of_iter = 500
    t = np.linspace(0, 500*env.unwrapped.dt, num__of_iter)


    plt.figure(layout="constrained", figsize=(15,7.5))
    for m in masa:
        local_thetas = []
        local_thetas_dot = []
        env.unwrapped.m = m
        print(m)
        env.unwrapped.state = np.array([np.pi,0])
        theta = 0
        theta_dot = 0
        for i in range(500):
            theta = np.arctan2(obs[1], obs[0])
            theta_dot = obs[2]
            local_thetas.append(theta)
            local_thetas_dot.append(theta_dot)
            theta_idx, theta_dot_idx = get_state_index((theta, theta_dot))
            u = policy[theta_idx, theta_dot_idx]
            obs, reward, done, truncated, info =  env.step([u])
        plt.subplot(2,1,1)
        plt.plot(t, local_thetas, label = f"m={m}")
        plt.subplot(2,1,2)
        plt.plot(t, local_thetas_dot, label = f"m={m}")

    plt.subplot(2,1,1)
    plt.legend()
    plt.title("θ(s)")
    plt.xlabel("t[s]")
    plt.ylabel("θ[rad]")

    plt.subplot(2,1,2)
    plt.legend()
    plt.title("θ_dot(s)")
    plt.xlabel("t[s]")
    plt.ylabel("θ_dot[rad/s]")

    plt.suptitle(f"thetas: {num_theta}, thetas_dot: {num_thetadot}, actions: {num_actions} , t: {eval_time}s", size=16)
    # plt.savefig(f"graphs/wyk_{num_theta}_{num_thetadot}_{num_actions}.pdf")
    # plt.close()
    plt.show()


thetas = [201]
theta_dots = [31]
actions=[11]

for theta in thetas:
    for theta_dot in theta_dots:
        for action in actions:
            testy(theta,theta_dot,action, [0.5, 1, 2])
            print("done")

