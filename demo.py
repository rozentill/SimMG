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

def func(x):
    y = np.sin(2*np.pi*x)*(4*np.pi**2+1)
    return y

def iterate_solve_Jacobian(A, f, N, max_iter=5000):

    iteration = max_iter
    

    L = np.tril(A, -1)
    U = np.triu(A, 1)
    
    D = np.diag(np.diag(A))
    R = np.linalg.inv(D).dot(-L-U)

    v_init = [0 for i in range(N-1)]
    v_init = np.expand_dims(v_init, 1)
    v_prev = v_init

    for i in range(iteration):

        v_curr = R.dot(v_prev) + np.linalg.inv(D).dot(f)
        v_prev = v_curr

    return v_curr

def direct_solve(A, f):

    return np.linalg.inv(A).dot(f)

def demo():
    
    N = 100
    h = 1./N
    
    A = discretization(N)

    f = np.array([func((i+1)*h) for i in range(N-1)])#boundary value 0
    f = np.expand_dims(f, 1)

    u = direct_solve(A, f)
    u = iterate_solve_Jacobian(A, f, N)

    plt.plot(f/(4*np.pi**2+1), label="exact solution")#u_true
    plt.plot(u, label="approximate solution")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    demo()