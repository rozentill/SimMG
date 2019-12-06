'''
Title: 1D simple multigrid solver for education
Author: Yuan Yao @ UBC Digital Geometry Group
Email: rozentill@gmail.com
'''

import numpy as np


# equation we are solving: -u''+u=f
# (-1/h^2)u_{i+1} + (2/h^2+1)u_{i} + (-1/h^2)u_{i-1} = f
def discretization(N):

    h = 1./N

    x = -1/h**2
    y =  (2/h**2 + 1)
    z = -1/h**2

    x = x*np.ones(N-1)
    y = y*np.ones(N)
    z = z*np.ones(N-1)

    A = np.diag(x, -1) + np.diag(y) + np.diag(z, 1)
    
    A[N-1,0] = -1/h**2
    A[0, N-1] = -1/h**2

    return A

def func(x):

	return x

def solve(A, f):

	pass


def demo():
    
    N = 5
    h = 1./N
    
    A = discretization(N)
    u_true = np.array([np.sin(np.pi*i*h) for i in range(N)])
    u_true = np.expand_dims(u_true, 1)
    print(u_true.shape)
    # f = func(np.array([i*h for i in range(N)]))
    f = A.dot(u_true)

    # u = solve(A,f)
    print(u_true)
    print(f)

if __name__ == '__main__':
    demo()