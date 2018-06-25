from math import cos, sin, tan
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import keyboard
import numpy as np
import random


p_true = [0.5, 0.5]

p = [0.01, 0.01]

dt = 0.1  # timestep

N = 100  # simulate for N timesteps

w = 5  # size of window

n = 10  # windows to visualized

delta_f_list = []
a_list = []

scoring_array = []

predicted_x = []
predicted_y = []


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


def main():
    # define initial state
    state = (0, 0, 0, 0)
    x = []
    y = []

    a = .1
    delta_f = .0

    delta_f_list.append(delta_f)
    a_list.append(a)

    states = [state]

    # simulate for N steps with constant controls
    for i in range(N):
        time.sleep(dt)
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

        delta_f_list.append(delta_f)
        a_list.append(a)

        # get derivative
        state_dot = f_prime(state, a, delta_f, p_true)

        # integrate
        # state = state + state_dot*dt
        x.append(state[0])
        y.append(state[1])

        state = euler(state, state_dot, dt)

        # print(state)
        # print('......................')
        states.append(state)

    states = np.array(states)

    x = np.array(x)
    y = np.array(y)

    windows = create_windows(states, w, delta_f_list, a_list, True)
    predicted_windows = predict_windows(windows, states)

    draw_windows(windows, predicted_windows)
    draw_tragetory(x,y, predicted_x, predicted_y)

def draw_tragetory(x,y,x_pred, y_pred, name = 'tragetory'):
    plt.plot(x_pred, y_pred, color='blue', label='predicted trajectory')
    plt.plot(x, y, color='red', label='actual trajectory')
    plt.legend(loc='best')
    plt.savefig(name)
    plt.close()

def draw_windows(windows, predicted_windows, name='windows'):
    x_actual, y_actual, x_pred, y_pred = extract_stages(
        windows, n, predicted_windows)

    x_actual_splits = [x_actual[i*w:(i+1)*w] for i in range(len(x_actual)//w)]
    y_actual_splits = [y_actual[i*w:(i+1)*w] for i in range(len(y_actual)//w)]

    x_pred_splits = [x_pred[i*w:(i+1)*w] for i in range(len(x_pred)//w)]
    y_pred_splits = [y_pred[i*w:(i+1)*w] for i in range(len(y_pred)//w)]

    for x_actual_val, y_actual_val, x_pred_val, y_pred_val in zip(x_actual_splits,
                                                                  y_actual_splits, x_pred_splits, y_pred_splits):
        plt.plot(x_actual_val, y_actual_val, color='red')
        plt.plot(x_pred_val, y_pred_val, color='blue')
    else:
        plt.plot(x_actual_val, y_actual_val,
                 color='red', label='actual windows')
        plt.plot(x_pred_val, y_pred_val, color='blue',
                 label='predicted windows')
    plt.legend(loc='best')
    plt.savefig(name)
    plt.close()



def extract_stages(windows, n, predicted_windows):
    x_actual = []
    y_actual = []
    x_pred = []
    y_pred = []

    for i in range(0, windows.shape[0]+1, n):
        # print(i)
        window = windows[i]
        window_pred = predicted_windows[i]

        states = [item[0] for item in window]
        states_pred = [item[0] for item in window_pred]

        [(x_actual.append(item[0]), y_actual.append(item[1]))
         for item in states]
        [(x_pred.append(item[0]), y_pred.append(item[1]))
         for item in states_pred]
    return x_actual, y_actual, x_pred, y_pred

    # print(scoring_array)
    # plt.show()


def predict_windows(windows, states):
    predicted_windows = []
    global predicted_x
    global predicted_y
    for i, window in enumerate(windows):
        # get first item in window list
        first_item_in_window = window[0]
        predicted_state = first_item_in_window[0]
        a = first_item_in_window[1]
        delta_f = first_item_in_window[2]

        predicted_states = [(predicted_state, a, delta_f)]

        predicted_x.append(predicted_state[0])
        predicted_y.append(predicted_state[1])

        rest_of_stages_in_window = window[1:]

        for state_and_parameters in rest_of_stages_in_window:
            predicted_state = state_and_parameters[0]
            a = state_and_parameters[1]
            delta_f = state_and_parameters[2]

            predicted_state_dot = f_prime(predicted_state, a, delta_f, p)

            predicted_x.append(predicted_state[0])
            predicted_y.append(predicted_state[1])
            predicted_state = euler(predicted_state, predicted_state_dot, dt)
            predicted_states.append([predicted_state, a, delta_f])

        predicted_states = np.array(predicted_states)
        # compute cost
        states_pred = predicted_states[:, 0]  # set all states
        # print(states_pred.shape)
        # print(state_considered.shape)
        # print('.................')
        scoring_array.append(cost_function(states_pred, states[i:w+i].T))

        predicted_windows.append(predicted_states)
    predicted_windows = np.array(predicted_windows)
    return predicted_windows


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


def create_windows(stages, n, delta_f_list, a_list, exact=True):
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
            window_states = stages[i:n+i]
            pos = i
            window = []
            for s in window_states:
                window.append([s, a_list[pos], delta_f_list[pos]])
                pos += 1
            windows.append(window)
    else:
        for i in range(shape[0] - n + 1):
            window_states = stages[i:n+i]
            pos = i
            window = []
            for s in window_states:
                window.append([s, a_list[pos], delta_f_list[pos]])
                pos += 1
            windows.append(window)
    windows = np.array(windows)
    return windows


if __name__ == '__main__':
    main()
