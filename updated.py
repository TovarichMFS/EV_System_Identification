'''
This is a sample program to help us test the optimization of our model
'''
# import tqdm
import numpy as np
from math import cos, tan, sin
import time
import random

from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


#globals
window_size = 40 #size of window
dt = 0.001   #turning angle -- delta
a = .1      #acceleration of the car
delta_f = .01 #delta_f

x = [0, 0, 0, 10] #starting position, acceleration and angle of the vehicle
p_true = [1.0, 1.0]  #true parameters of the car

def f_prime(state, a, delta_f, p):
    '''
    This function computes the derivative of a function 
    INPUT:
        state: a numpy.ndarray vector of type 'float' containing values represents f(t)
        a: float representing acceleration
        delta_f: float representing delta
        p: List of float representing parameters of car
    OUTPUT:
        state_dot: a numpy.ndarray vector of type 'float' containing values represents f'(t)
    '''
    x = state[0]  # x-coordinates
    y = state[1]  # y- coordinates
    psi = state[2] # turning angle
    v = state[3] # velocity of car

    lr = p[0]   #distance from middle of the car to the left wheel
    lf = p[1]   #distance from middle of the car to the right wheel

    if abs(delta_f) > 0:
        beta = 1/tan(lr/(lf + lr)*tan(delta_f))
    else:
        beta = 0

    x_dot = v*cos(psi+beta)
    y_dot = v*sin(psi+beta)

    psi_dot = v/lr*sin(beta)

    #compute next state of the car
    state_dot = np.array([x_dot, y_dot, psi_dot, a])

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
    return np.array(result)

def cost_function(actual_stages, predicted_states):
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

def create_window(window_size, p, x0, dt , a , delta_f, nparray = True):
    '''
    This function creates windows each of size win_size items using parameter p, initial point x0 and delta dt
    INPUT:
        win_size : int representing the size of the window
        p : i-d representing the parameters of the car
        x0: 1-d array containing the starting stage of the car
        df: float representing delta
        nparray : boolean that indicate if to return a window as numpy array or python array, True by default
    OUTPUT:
        result : ndarray or python array of stages representing trajectory of the car.
    '''
    results = [x0]

    for i in range(window_size):
        x = results[-1]
        dx = f_prime(x, a, delta_f, p)
        x = euler(x, dx, dt)
        results.append(x)
    if nparray:
        results = np.array(results)
    return results


def compute_residual(p):
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
    observed = create_window(window_size, p_true, x, dt, a, delta_f)
    predicted = create_window(window_size, p, x, dt, a, delta_f)
    cost = cost_function(observed, predicted)
    return cost


def optimize_residual_function(residual_func, initial_guess):
    '''
    This function optimizes our cost function so as to adjust the predicted trajectory to the actual trajectory. 
    INPUT:
        func : Function which computes the vector of residuals, with the signature fun(p, *args, **kwargs)
        initial_guess : Initial guess of parameters usually numpy 1-d ndarray 
    OUTPUT:
        result: OptimizeResult from the scipy least square optimization
    '''
    result = least_squares(residual_func, initial_guess )
    return result

def generate_guesses(shape):
    '''
    This function generates a random numpy array of shape provided
    INPUT:
        shape : Shape of nparray to create with random values
    OUTPUT:
        array : nparray of shape shape filled with random values
    '''
    return np.random.random(shape)

def main():
    '''
    Sample Test for our implementation with a single windows 
    This is a test function that simulates a trajectory, 
    makes a prediction on the trajectory, optimized the cost between 
    the two and finally plot both trajectories.
    '''
    p0 = [3.0, 3.0]  #predicted parameters of the car

    observed = create_window(window_size, p_true, x, dt, a, delta_f)
    predicted = create_window(window_size, p0, x, dt, a , delta_f)

    # print(observed - predicted)

    # initial_guess = np.array([0.2, 0.4])
    initial_guess = generate_guesses((2,))

    optimized_model = optimize_residual_function(compute_residual, initial_guess)

    # predicted_model = create_window(window_size, optimized_model.x, x, dt, a, delta_f)

    plt.plot(observed[:, 0], observed[:, 1], label='observed model')
    plt.plot(predicted[:, 0], predicted[:, 1], label='predicted')
    plt.legend(loc='best')
    plt.show()


def test(num_of_windows):
    '''
    Sample Test for our implementation with multiple windows 
    '''
    global x
    observed = []
    predicted = []
    for i in range(num_of_windows):
        n = random.randint(1, 6) # random p0 values
        p0 = [n, n] 
        observed.extend(create_window(window_size, p_true, x, dt, a, delta_f))
        predicted.extend(create_window(window_size, p0, x, dt, a , delta_f))
        x = observed[-1]  #set the last state as new stage

    observed = np.array(observed)
    predicted = np.array(predicted)

    # initial_guess = np.array([0.2, 0.4])
    initial_guess = generate_guesses((2,))
    print(optimize_residual_function(compute_residual, initial_guess))

    plt.plot(observed[:, 0], observed[:, 1], label='observed model')
    plt.plot(predicted[:, 0], predicted[:, 1], label='predicted')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    # main()  #Test with a single window
    test(10)    #Test with multiple windows 
