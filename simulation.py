from math import cos, sin, tan
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keyboard
import numpy as np
import random

lr = 0.5  # distance from CoM to rear wheels
lf = 0.5  # distance from CoM to front wheels

dt = 0.1  # timestep

N = 50  # simulate for 1000 timesteps

w = 5  # size of window


def f_prime(state, a, delta_f):
    '''
    This function computes the prime of a function 
    INPUT:
        state: a numpy.ndarray vector of type 'float' containing values represents f(t)
        a: float representing acceleration
        dt: float representing delta
    OUTPUT:
        state_dot: a numpy.ndarray vector of type 'float' containing values represents f'(t)
    '''
    x = state[0]
    y = state[1]
    psi = state[2]
    v = state[3]

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

    result = np.array([0.0] * len(state))

    result[0] = state[0] + state_dot[0]*dt
    result[1] = state[1] + state_dot[1]*dt
    result[2] = state[2] + state_dot[2]*dt
    result[3] = state[3] + state_dot[3]*dt

    return result

def main():
    # define initial state
    state = np.array([0, 0, 0, 0])
    random_state = np.array([0, 0, 0, 0]) 
    x = []
    y = []

    random_x = []
    random_y = []

    a = .1
    delta_f = .0

    states = [state]
    random_states = [random_state]
    # simulate for N steps with constant controls
    for i in range(N):

        if i % w == 0: # if window
            random_state = state
        time.sleep(0.1)
        if keyboard.is_pressed('q'):
            print('left')
            delta_f += 0.0001
        elif keyboard.is_pressed('z'):
            print('up')
            a += 0.01
        elif keyboard.is_pressed('d'):
            print('right')
            delta_f -= 0.0001
        elif keyboard.is_pressed('s'):
            print('down')
            a -= 0.01
        else:
            if delta_f > 0.0:
                delta_f -= 0.0001
            elif delta_f < 0.0:
                delta_f += 0.0001

        # get derivative
        state_dot = f_prime(state, a, delta_f)
        random_state_dot = f_prime(random_state, random.uniform(-1,1), random.uniform(-1, 1))

        # integrate
        # state = state + state_dot*dt
        x.append(state[0])
        y.append(state[1])
        random_x.append(random_state[0])
        random_y.append(random_state[1])

        state = euler(state, state_dot, dt)
        random_state = euler(random_state,random_state_dot, dt)
        print(state)
        print(random_state)
        print('......................')
        states.append(state)
        random_states.append(random_state)


    states = np.array(states)
    random_states = np.array(random_states)

    random_x = np.array(random_x)
    random_y = np.array(random_y)
    x = np.array(x)
    y = np.array(y)

    plt.plot(random_x,random_y)
    plt.plot(x,y)
    plt.savefig('myfig')
    # plt.show()

    #testing cost function
    print('cost value for x"s is {}'.format(cost_function(x,random_x)))
    print('cost value for y"s is {}'.format(cost_function(y,random_y)))
    print('cost value for states is {}'.format(cost_function(states,random_states)))

    # print resultng trajectory
    # TODO: add visualization
    """for s in states:
        print(s)"""


def window_stack(array, stepsize, width):
    '''
    This function creates a window stack from the array provided
    INPUT:
        array: a numpy.ndarray vector of type 'float' containing values
        stepsize: float representing step size of the window
        width : int representing the width of the width
    OUTPUT:
        ndarray of values divided into windows stacks
    '''
    n = array.shape[0]
    return np.hstack( array[i:1+n+i-width:stepsize] for i in range(0,width))

def last_window(array, n):
    '''
    This function returns the last nth item in an array
    INPUT:
        array: a numpy.ndarray vector of type 'float' containing values
        size : int representing the starting index of the window
    OUTPUT:
        ndarray of last nth items in the array
    '''
    return array[-n:]


def cost_function(correct_stage, predicted_stage):
    '''
    This function calculates the Mean Squared Error given two sets of output values, 
    one set corresponding to the correct values, the other set 
    representing the output values predicted by a regression model
    INPUT:
        predicted_stage: a numpy.ndarray vector of type 'float' containing m predicted values
        correct_stage: a numpy.ndarray vector of type 'float' containing m correct values
    OUTPUT:
        err: 'float' representing the Mean Squared Error
    '''
    err = np.mean(np.square(correct_stage - predicted_stage))
    return err

def create_windows(stages, n, exact=False):
    '''
    This function creates windows each of size n items from items in stages
    INPUT:
        stages : ndarray containing stages of car
        n : int representing maximum number of stages each window can contain
        exact: boolean requesting windows to have strictly n number of stages
    OUTPUT:
        windows : ndarray of windows each contains a maximum of n stages
    '''
    shape = stages.shape
    windows = []
    if not exact:
        for i in range(shape[0]):
            windows.append(stages[i:n+i])
    else:
        for i in range(shape[0] - n + 1):
            windows.append(stages[i:n+i])
    windows = np.array(windows)
    return windows


if __name__ == '__main__':
    #stages = np.arange(0, 12)
    stages = np.random.randn(*(7,4))
    print(stages)
    windows = create_windows(stages, 3, False)
    print('...............')
    for i in windows:
        print(i)

    #main()
    # l = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]]
    # a = np.array(l)
    # # # print(a)
    # # slide_window = window_stack(a, 1, 4)
    # # print(slide_window[0])
    # arr = last_window(a, 4)
    # print(arr)
