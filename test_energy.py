import numpy as np
from numba import jit
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


theta_min, theta_max = -np.pi, np.pi
thetadot_min, thetadot_max = -8, 8
g = 9.81
l=1
m = 1
dt = 0.01

@jit()
def get_next_state(state: list[float, float], action):
    
    theta, thetadot = state
    u = action

    # reward = - ((theta)**2 + 0.1 * (thetadot**2) + 0.001 * (u**2))
    newthdot = thetadot + (3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l**2) * u) * dt

    if newthdot < thetadot_min:
        newthdot = thetadot_min
    elif newthdot > thetadot_max:
        newthdot = thetadot_max
    next_theta = theta + newthdot * dt

    # Ensure the angle stays within the range [-pi, pi]
    # next_theta = (next_theta + np.pi) % (2 * np.pi) - np.pi

    return [next_theta, newthdot]



x = [[np.pi/2, 0.0]]
end_t = 1000
for i in range(int(end_t/dt)):
    x_new = get_next_state(x[-1], 0)
    x.append(x_new)


X = np.array(x)

E_k = 0.5 * m * X[:, 1]* X[:,1]
E_p = m * g * np.cos(X[:,0])
E = E_k + E_p

T = np.linspace(0, end_t, int(end_t/dt)+1)

plt.plot(T, X[:,1])
plt.show()

plt.plot(T, E_k, label = "E_k")
plt.plot(T, E_p, label = "E_p")
plt.plot(T, E, label = "E")
plt.xlabel("t[s]")
plt.ylabel("E[J]")
plt.suptitle("E(t) (gymnasium approximation)")
plt.legend()
plt.show()





# import numpy as np
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt
# import pandas as pd

# # Stałe
# g = 9.81  # przyspieszenie grawitacyjne [m/s^2]
# L = 1.0   # długość nici [m]

# # Funkcja opisująca równanie różniczkowe wahadła matematycznego
# def pendulum_ode(t, y):
#     theta, omega = y
#     dydt = [omega, (g / L) * np.sin(theta)]
#     return dydt

# # Nowy kąt początkowy: 0 oznacza pozycję pionową w górę
# theta0_new = np.pi / 2 

# # Warunki początkowe: kąt początkowy i prędkość kątowa
# omega0 = 0.0  # początkowa prędkość kątowa
# y0_new = [theta0_new, omega0]


# for solver in ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']:
#     # Przedział czasowy
#     dt = 0.001
#     t_span = (0, 1000)
#     t_eval = np.linspace(t_span[0], t_span[1], int(t_span[1]/dt))

#     # Rozwiąż równanie różniczkowe z nowymi warunkami początkowymi
#     sol_new = solve_ivp(pendulum_ode, t_span, y0_new, t_eval=t_eval, method=solver)

#     # Wykres wyników
#     plt.figure(figsize=(10, 6))
#     plt.plot(sol_new.t, sol_new.y[0], label='θ')
#     plt.plot(sol_new.t, sol_new.y[1], label='ω')
#     plt.xlabel('t[s]')
#     plt.ylabel('')
#     plt.legend()
#     plt.title(f'θ[rad], ω[rad/s] {solver}')
#     plt.grid(True)
#     plt.savefig(f"graphs/theta_omega_{solver}.pdf")
#     plt.close()
#     X = np.array(sol_new.y).transpose()
#     E_k = 0.5 * m * X[:, 1]* X[:,1]
#     E_p = m * g * np.cos(X[:,0])
#     E = E_k + E_p

#     plt.plot(sol_new.t, E_k, label = "E_k")
#     plt.plot(sol_new.t, E_p, label = "E_p")
#     plt.plot(sol_new.t, E, label = "E")
#     plt.xlabel("t[s]")
#     plt.ylabel("E[J]")
#     plt.suptitle(f"E(t) {solver}")
#     plt.legend()
#     plt.savefig(f"graphs/E_{solver}.pdf")
#     plt.close()