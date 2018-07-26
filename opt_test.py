'''
This is a sample program to help us test the optimization of our model
'''
# import tqdm
import numpy as np
from math import *
import time

from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def f_prime(state, a, delta_f, p):
    '''
    This function computes the derivative of a function 
    INPUT:
        state: a numpy.ndarray vector of type 'float' containing values represents f(t)
        a: float representing acceleration
        delta_f: float representing delta
        p: List of float representing parameters of vehicle
    OUTPUT:
        state_dot: a numpy.ndarray vector of type 'float' containing values represents f'(t)
    '''
    x = state[0]
    y = state[1]
    psi = state[2]
    v = state[3]

    lr = p[0]
    lf = p[1]

    if abs(delta_f) > 0:
        beta = 1/tan(lr/(lf + lr)*tan(delta_f))
    else:
        beta = 0

    x_dot = v*cos(psi+beta)
    y_dot = v*sin(psi+beta)

    psi_dot = v/lr*sin(beta)
    v_dot = a

    state_dot = np.array([x_dot, y_dot, psi_dot, v_dot])

    return state_dot


# implements f(t+1) = f(t) + f'(t)*dt
def euler(state, state_dot, dt):
    '''
    This function evaluates the euler value given a state, state_dot and delta.
    It evaluates f(t+1) = f(t) + f'(t)*dt
    INPUT:
        state: a numpy.ndarray vector of type 'float' containing values represents f(t)
        state_dot: a numpy.ndarray vector of type 'float' containing values represents f'(t)
        dt: float representing delta
    OUTPUT:
        result: ndarray of euler values
    '''
    result = state + state_dot * dt
    return result

def cost_function(actual_stages, predicted_states):
    # print('shape of actual_stage is {} and predicted is {}'.format(actual_stages.shape, predicted_states.shape))
    '''
    This function calculates the Mean Squared Error given two sets of output values, 
    one set corresponding to the correct values, the other set 
    representing the output values predicted by a regression model
    INPUT:
        predicted_states: a numpy.ndarray vector of type 'float' containing m predicted values
        actual_stages: a numpy.ndarray vector of type 'float' containing m correct values
    OUTPUT:
        err: 'float' representing the Mean Squared Error
    '''
    n = actual_stages.shape[0]
    constant = 1 / (2 * n)
    diff = 0.0
    for i in range(n):
        diff += np.mean(np.square(actual_stages[i] - predicted_states[i]))
    cost = constant * diff
    return cost


def create_window(win_size, p, x0, dt):
    '''
    This function creates windows each of size win_size items using parameter p, initial point x0 and delta dt
    INPUT:
        win_size : int representing the size of the window
        p : i-d representing the parameters of the vehicle
        x0: 1-d array containing the starting stage of the vehicle
        df: float representing delta
    OUTPUT:
        result : ndarray of stages representing trajectory of the vehicle.
    '''
    result = [x0]

    for i in range(window_size):
        x = result[-1]
        dx = f_prime(x, 0, 0.01, p)
        x = euler(x, dx, dt)
        result.append(x)

    return np.array(result)


def func(p):
    '''
    Function which computes the vector of residuals, with the signature fun(p, *args, **kwargs), 
    i.e., the minimization proceeds with respect to its first argument. The argument p passed to 
    this function is an ndarray of shape (n,) (never a scalar, even for n=1). It must return a 
    1-d array_like of shape (m,) or a scalar. If the argument p is complex or the function fun 
    returns complex residuals.
    This function predicts the window using parameter p and computes the cost of the predicted window.
    INPUT:
        p: passed to this function is an ndarray of shape (n,)
    OUTPUT:
        cost: 'float' representing the Mean Squared Error (cost of predicted and actual)
    '''
    predicted = create_window(window_size, p, x, dt)
    cost = cost_function(observed, predicted)
    return cost

def optimize_model(func, initial_guess):
    '''
    This function optimizes our cost function so as to adjust the predicted trajectory to the actual trajectory. 
    INPUT:
        func : Function which computes the vector of residuals, with the signature fun(p, *args, **kwargs)
        initial_guess : Initial guess of parameters usually numpy 1-d ndarray 
    OUTPUT:
        result: OptimizeResult from the scipy least square optimization
    '''
    result = least_squares(func, initial_guess )
    return result


p_true = [1.0, 1.0]

p0 = [3.0, 3.0]

window_size = 40
dt = 0.001

x = [0, 0, 0, 10]
observed = create_window(window_size, p_true, x, dt)
predicted = create_window(window_size, p0, x, dt)

#initial_guess = [0.2, 0.3,4,5]
initial_guess = np.array([0.2, 0.4])

observed = np.array(observed)
predicted = np.array(predicted)

optimized_model = optimize_model(func, initial_guess)

predicted_model = create_window(window_size, optimized_model.x, x, dt)

# cost_function(observed, predicted)
# print(predicted_model)
print(optimized_model)
# xx = optimized_model.x
# yy = optimized_model.grad
plt.plot(observed[:, 0], observed[:, 1], label='observed model')
plt.plot(predicted[:, 0], predicted[:, 1], label='predicted')
plt.plot(predicted_model[:, 0], predicted_model[:, 1], label='optimized model')
# plt.plot(predicted_model[:,0], predicted_model[:,1])
plt.legend(loc='best')
# plt.plot(yy)
plt.show()


# print(cost_function(observed, predicted))





# steps_p0 = 10
# steps_p1 = 10

# grid_p0 = []
# grid_p1 = []
# value = []

# dp0 = 0.01
# dp1 = 0.01


# for i in range(steps_p0):
#     for j in range(steps_p1):
#         grid_p0.append(dp0*i + 0.5)
#         grid_p1.append(dp1*j + 0.5)
#         # print(grid_p0[-1])
#         # print(grid_p1[-1])
#         model = optimize_model(func, [grid_p0[-1], grid_p1[-1]])
#         print(model)
#         time.sleep(1)
        
        # value.append(func([grid_p0[-1], grid_p1[-1]]))



# print(len(value))
# plt.plot(grid_p1)
# plt.show()
# fig = plt.figure()
# ax = fig.gca(projection='3d')

# surf = ax.plot_surface(np.array(grid_p0), np.array(grid_p1), np.array(value),
#                        linewidth=0, antialiased=False)

# plt.show()




# print(least_squares(func, p0))

