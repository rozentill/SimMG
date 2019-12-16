'''
Title: 1D simple multigrid solver for education
Author: Yuan Yao @ UBC Digital Geometry Group
Email: rozentill@gmail.com
'''

import numpy as np
import matplotlib.pyplot as plt

# equation we are solving: -u''+u=f
# (-1/h^2)u_{i+1} + (2/h^2+1)u_{i} + (-1/h^2)u_{i-1} = f_{i}
def discretization(N):

    h = 1./N

    x = -1/(h**2)
    y =  (2/h**2 + 1)
    z = -1/(h**2)

    x = x*np.ones(N-2)
    y = y*np.ones(N-1)
    z = z*np.ones(N-2)

    A = np.diag(x, -1) + np.diag(y) + np.diag(z, 1)

    return A

def exact_solution(x):
    y = np.sin(2*np.pi*x)
    return y

def func(x):
    y = np.sin(2*np.pi*x)*(4*np.pi**2+1)
    return y

def init_func(x):
    y = np.sin(8*np.pi*x)*(4*np.pi**2+1)
    return y

def iterate_solve_GaussSeidel(A, f, N, u_true, max_iter=100):
    h=1./N
    iteration = max_iter
    
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    
    D = np.diag(np.diag(A))
    R = np.linalg.inv(D+L).dot(-U)

    v_init = [init_func((i+1)*h) for i in range(N-1)]
    v_init = np.expand_dims(v_init, 1)
    v_prev = v_init
    error = []
    for i in range(iteration):

        v_curr = R.dot(v_prev) + np.linalg.inv(D+L).dot(f)
        v_prev = v_curr
        error.append(np.linalg.norm(v_curr-u_true, np.inf))

    return v_curr, error

def iterate_solve_Jacobian(A, f, N, max_iter=100):

    iteration = max_iter
    

    L = np.tril(A, -1)
    U = np.triu(A, 1)
    
    D = np.diag(np.diag(A))
    R = np.linalg.inv(D).dot(-L-U)

    v_init = [init_func((i+1)*h) for i in range(N-1)]
    v_init = np.expand_dims(v_init, 1)
    v_prev = v_init

    for i in range(iteration):

        v_curr = R.dot(v_prev) + np.linalg.inv(D).dot(f)
        v_prev = v_curr

    return v_curr

# return result of Au=f 
def V_cycle_at_h(A_h, f_h, N_h):

    if N_h == 2:
    
        return (1./A_h).dot(f_h) # not use inverse


    # iterpolation: from coarse to fine, (n-1, n/2-1)
    interpolate_h = np.zeros((N_h-1, N_h//2-1))
    for i in range(0, N_h//2-1):
        start = 2*i
        interpolate_h[start, i] = 1./2
        interpolate_h[start+1, i] = 1
        interpolate_h[start+2, i] = 1./2
    
    # restriction operator
    restriction_h = interpolate_h.transpose()/4.

    L = np.tril(A_h, -1)
    U = np.triu(A_h, 1)
    
    D = np.diag(np.diag(A_h))
    R = np.linalg.inv(D).dot(-L-U)

    v_init = [0 for i in range(N_h-1)]
    v_init = np.expand_dims(v_init, 1)
    v_h = R.dot(v_init) + np.linalg.inv(D).dot(f_h)
    r_h = f_h - A_h.dot(v_h)

    r_2h = restriction_h.dot(r_h)
    A_2h = restriction_h.dot(A_h.dot(interpolate_h))

    e_2h = V_cycle_at_h(A_2h, r_2h, N_h//2)
    e_h = interpolate_h.dot(e_2h)
    v_h += e_h

    return v_h

def iterate_solve_multigrid(A, f, N, u_true, max_iter=16):
    
    # hyperparameters
    iter_pre = 1
    iter_post = 1
    h=1./N

    A_h = A

    # # iterpolation: from coarse to fine, (n-1, n/2-1)
    interpolate_h = np.zeros((N-1, N//2-1))
    for i in range(0, N//2-1):
        start = 2*i
        interpolate_h[start, i] = 1./2
        interpolate_h[start+1, i] = 1
        interpolate_h[start+2, i] = 1./2
    
    # restriction operator
    restriction_h = interpolate_h.transpose()/4.

    #initilize 
    L = np.tril(A_h, -1)
    U = np.triu(A_h, 1)
    
    D = np.diag(np.diag(A_h))
    R = np.linalg.inv(D).dot(-L-U)

    v_init = [0 for i in range(N-1)]
    v_init = np.expand_dims(v_init, 1)
    v_prev = v_init
    error = []
    for iter in range(max_iter):

        # pre-iteration
        for i in range(iter_pre):
            
            v_curr = R.dot(v_prev) + np.linalg.inv(D).dot(f)
            v_prev = v_curr

        r_h = f - A_h.dot(v_curr)
        r_2h = restriction_h.dot(r_h)
        A_2h = restriction_h.dot(A_h.dot(interpolate_h))

        e_2h = V_cycle_at_h(A_2h, r_2h, N//2)
        # e_2h = np.linalg.inv(A_2h).dot(r_2h)
        e_h = interpolate_h.dot(e_2h)
        v_curr = v_curr + e_h
        v_prev = v_curr

        # post-iteration
        for i in range(iter_post):
            
            v_curr = R.dot(v_prev) + np.linalg.inv(D).dot(f)
            v_prev = v_curr

        error.append(np.linalg.norm(v_curr[:, 0]-u_true, np.inf))

    return v_curr, error

def direct_solve(A, f):

    return np.linalg.inv(A).dot(f)

def demo():
    for n in range(4):
        N = 2**(n+4)
        h = 1./N
        
        A = discretization(N)

        f = np.array([func((i+1)*h) for i in range(N-1)])#boundary value 0
        f = np.expand_dims(f, 1)

        u_true = np.array([exact_solution((i+1)*h) for i in range(N-1)])
        u, e = iterate_solve_multigrid(A, f, N, u_true)

        plt.plot(e)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    demo()