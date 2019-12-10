'''
Title: 1D simple multigrid solver for education
Author: Yuan Yao @ UBC Digital Geometry Group
Email: rozentill@gmail.com
'''

import numpy as np
import matplotlib.pyplot as plt

# equation we are solving: -u''+u=f
# (-1/h^2)u_{i+1} + (2/h^2+1)u_{i} + (-1/h^2)u_{i-1} = f
def discretization(N):

    h = 1./N

    x = -1/(h**2)
    y =  (2/h**2 + 1)
    z = -1/(h**2)

    x = x*np.ones(N-2)
    y = y*np.ones(N-1)
    z = z*np.ones(N-2)

    A = np.diag(x, -1) + np.diag(y) + np.diag(z, 1)
    
    # A[N-1,0] = -1/h**2
    # A[0, N-1] = -1/h**2

    return A

def func(x):

	return x

def solve(A, f):

	pass


def demo():
    
    N = 4
    h = 1./N
    
    A = discretization(N)
    u_true = np.array([np.sin(2*np.pi*(i+1)*h) for i in range(N-1)])
    u_true = np.expand_dims(u_true, 1)
    print(u_true.shape)
    # f = func(np.array([i*h for i in range(N)]))
    f = A.dot(u_true)
    f_true = u_true*(4*np.pi**2+1)
    # u = solve(A,f)
    print(u_true)
    print(f)
    print(A)
    print(max(f))
    print(max(f_true))
    plt.plot(f_true)
    plt.plot(f)
    plt.show()
if __name__ == '__main__':
    demo()