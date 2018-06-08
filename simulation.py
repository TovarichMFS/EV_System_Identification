from math import cos, sin, tan
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keyboard
import numpy as np

lr = 0.5  # distance from CoM to rear wheels
lf = 0.5  # distance from CoM to front wheels

dt = 0.1  # timestep

N = 300  # simulate for 1000 timesteps

def f_prime(state, a, delta_f):
    x   = state[0]
    y   = state[1]
    psi = state[2]
    v   = state[3]

    if abs(delta_f) > 0:
        beta = 1/tan(lr/(lf + lr)*tan(delta_f))
    else:
        beta = 0

    x_dot = v*cos(psi+beta)
    y_dot = v*sin(psi+beta)

    psi_dot = v/lr*sin(beta)
    v_dot = a


    state_dot = [x_dot, y_dot, psi_dot, v_dot]

    return state_dot


# implements f(t+1) = f(t) + f'(t)*dt
def euler(state, state_dot, dt):
    result = [0.0] * len(state)

    result[0] = state[0] + state_dot[0]*dt
    result[1] = state[1] + state_dot[1]*dt
    result[2] = state[2] + state_dot[2]*dt
    result[3] = state[3] + state_dot[3]*dt

    return result

def main():
    # define initial state
    state = [0, 0, 0, 0] 
    x = []
    y = []

    a = .1
    delta_f = .0

    states = [state]

    # simulate for N steps with constant controls
    for i in range(N):
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

        # integrate
        # state = state + state_dot*dt
        x.append(state[0])
        y.append(state[1])
        state = euler(state, state_dot, dt)
        print(state)
        states.append(state)
    plt.plot(x,y)
    plt.savefig('myfig')
    plt.show()

    # print resultng trajectory
    # TODO: add visualization
    """for s in states:
        print(s)"""


def window_stack(array, stepsize, width):
    n = array.shape[0]
    return np.hstack( array[i:1+n+i-width:stepsize] for i in range(0,width) )

def last_window(array, size):
    return array[-size:]

if __name__ == '__main__':
    main()
    # l = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]]
    # a = np.array(l)
    # # # print(a)
    # # slide_window = window_stack(a, 1, 4)
    # # print(slide_window[0])
    # arr = last_window(a, 4)
    # print(arr)
