from math import *

lr = 0.5  # distance from CoM to rear wheels
lf = 0.5  # distance from CoM to front wheels

dt = 0.1  # timestep

N = 1000  # simulate for 1000 timesteps

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

    a = .1
    delta_f = .0

    states = [state]

    # simulate for N steps with constant controls
    for i in range(N):
        # get derivative
        state_dot = f_prime(state, a, delta_f)

        # integrate
        # state = state + state_dot*dt

        state = euler(state, state_dot, dt)
        states.append(state)

    # print resultng trajectory
    # TODO: add visualization
    for s in states:
        print(s)




if __name__ == '__main__':
    main()



    
