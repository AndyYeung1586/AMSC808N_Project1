# Tak Yeung
# AMSC808N Project 1
import numpy as np
import scipy
import matplotlib.pyplot as plt


def main():
    # Problem 5
    P5()

    # Problem 6
    P6()
    return


def P5():
    # ================================= Problem 5 =================================
    def u_exact(x):
        return np.cos(np.pi * x) + 1

    def g(x, scale):
        u4 = np.pi ** 4 * np.cos(np.pi * x) * scale**4
        u2 = -np.pi ** 2 * np.cos(np.pi * x) * scale**2
        return u4 - 4 * u2 + 3 * u_exact(x)

    def b(x, x0, x1, scale):
        """ boundary function """
        return 0

    def bp(x, x0, x1, scale):
        """ b' """
        return 0

    # Solve problem 5 on the Chebyshev extrema grid
    zmin = -1
    zmax = 1
    Nx = 1000
    x_eval = np.linspace(-1, 1, Nx)
    z = np.linspace(zmin, zmax, Nx)
    sol_list = []

    nlist = [5, 10, 20, 40]
    N_list = len(nlist)
    for n in nlist:
        # construct L operator
        D, x = cheb(n)
        D2 = np.matmul(D, D)
        D4 = np.matmul(D2, D2)
        L = D4 - 4*D2 + 3*np.eye(n+1)
        q_out = solve_BVP1(n, zmin, zmax, L, g, b, bp, x_eval)
        sol_list.append(q_out)

    # plot the solution
    fig_size = (11, 9)
    fig, ax = plt.subplots(figsize=fig_size)
    plt.plot(z, u_exact(z), linewidth=3, label='Exact Solution')
    for j in range(N_list):
        q = sol_list[j]
        plt.plot(z, q, label=f"N = {nlist[j]}")
    plt.xlabel("z")
    plt.ylabel("q_out")
    ax.legend(loc='upper right')
    plt.title("Numerical solution to problem 5")
    plt.grid()
    plt.show()

    # plot the absolute error
    fig, ax = plt.subplots(figsize=fig_size)
    q_ref = sol_list[-1]
    for j in range(N_list - 1):
        q = sol_list[j]
        plt.plot(z, abs(q - q_ref), label=f"N = {nlist[j]}")
    plt.xlabel("z")
    plt.ylabel("The absolute error")
    ax.set_yscale('log')
    ax.legend(loc='upper left')
    plt.title("Numerical errors of the chebyshev spectral method")
    plt.show()

    # plot max error vs. degree
    NE = 50
    step = 2
    nmin = 4
    E_cheb_int_ecos = np.zeros((NE,))
    deg = np.zeros((NE,), dtype=int)
    for n in range(NE):
        # construct L operator
        deg[n] = nmin + n * step
        D, x = cheb(deg[n])
        D2 = np.matmul(D, D)
        D4 = np.matmul(D2, D2)
        L = D4 - 4 * D2 + 3 * np.eye(deg[n] + 1)
        q_out = solve_BVP1(deg[n], zmin, zmax, L, g, b, bp, x_eval)
        E_cheb_int_ecos[n] = np.max(np.abs(q_out - u_exact(z)))

    # plot errors
    fig, ax = plt.subplots(figsize=fig_size)
    plt.plot(deg, E_cheb_int_ecos, '-v', label="cos(pi*x)+1")
    plt.xlabel("Degree of Chebyshev polynomial")
    plt.ylabel("Max error")
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    plt.title("Errors Decay")
    plt.grid()
    plt.show()
    return


def P6():
    # ================================= Problem 5 =================================
    def u_ex(x):
        return 1./(1+x**2)

    def ux_ex(_x):
        return -2*_x/(1+_x**2)**2

    def g(x, scale):
        u4 = (120.*x**4 - 240*x**2 + 24)/(1+x**2)**5
        u1 = -2.*x/(1+x**2)**2
        u0 = 1./(1+x**2)
        return scale**4 * u4 + scale*u1 + u0

    def b(x, x0, x1, scale):
        """ boundary function """
        p0 = 0.25 * (x - 1.0) ** 2 * (x + 2.0)
        q0 = 0.25 * (x - 1.0) ** 2 * (x + 1.0)
        p1 = 0.25 * (x + 1.0) ** 2 * (2.0 - x)
        q1 = 0.25 * (x + 1.0) ** 2 * (x - 1.0)

        u0 = u_ex(x0)
        up0 = ux_ex(x0) * scale
        u1 = u_ex(x1)
        up1 = ux_ex(x1) * scale
        return p0 * u0 + q0 * up0 + p1 * u1 + q1 * up1

    def bp(x, x0, x1, scale):
        """ derivative of b """
        u0 = u_ex(x0)
        up0 = ux_ex(x0) * scale
        u1 = u_ex(x1)
        up1 = ux_ex(x1) * scale
        return u0 * (3 * x ** 2 - 3) / 4 + up0 * (3 * x ** 2 - 2 * x - 1) / 4 + u1 * (-3 * x ** 2 + 3) / 4 + up1 * (
                    3 * x ** 2 + 2 * x - 1) / 4

    # Solve problem 5 on the Chebyshev extrema grid
    zmin = 0
    zmax = 5
    Nx = 1000
    x_eval = np.linspace(-1, 1, Nx)
    z = np.linspace(zmin, zmax, Nx)
    sol_list = []
    nlist = [5, 10, 20, 40, 80]
    N_list = len(nlist)
    for n in nlist:
        # construct L operator
        D, x = cheb(n)
        D4 = np.matmul(np.matmul(np.matmul(D, D), D), D)
        L = D4 + D + np.eye(n+1)
        q_out = solve_BVP1(n, zmin, zmax, L, g, b, bp, x_eval)
        sol_list.append(q_out)

    # plot the solution
    fig_size = (11, 9)
    fig, ax = plt.subplots(figsize=fig_size)
    plt.plot(z, u_ex(z), linewidth=3, label='Exact Solution')
    for j in range(N_list):
        q = sol_list[j]
        plt.plot(z, q, label=f"N = {nlist[j]}")
    plt.xlabel("z")
    plt.ylabel("q_out")
    ax.legend(loc='upper right')
    plt.title("Numerical solution to problem 6")
    plt.grid()
    plt.show()

    # plot the absolute error
    fig, ax = plt.subplots(figsize=fig_size)
    q_ref = sol_list[-1]
    for j in range(N_list - 1):
        q = sol_list[j]
        plt.plot(z, abs(q - q_ref), label=f"N = {nlist[j]}")
    plt.xlabel("z")
    plt.ylabel("The absolute error")
    ax.set_yscale('log')
    ax.legend(loc='upper left')
    plt.title("Numerical errors of the chebyshev spectral method")
    plt.show()

    # plot max error vs. degree
    NE = 50
    step = 2
    nmin = 4
    E_cheb_int_ecos = np.zeros((NE,))
    deg = np.zeros((NE,), dtype=int)
    for n in range(NE):
        # construct L operator
        deg[n] = nmin + n * step
        D, x = cheb(deg[n])
        D4 = np.matmul(np.matmul(np.matmul(D, D), D), D)
        L = D4 + D + np.eye(deg[n] + 1)
        q_out = solve_BVP1(deg[n], zmin, zmax, L, g, b, bp, x_eval)
        E_cheb_int_ecos[n] = np.max(np.abs(q_out - u_ex(z)))

    # plot errors
    fig, ax = plt.subplots(figsize=fig_size)
    plt.plot(deg, E_cheb_int_ecos, '-v', label="compare to exact")
    plt.xlabel("Degree of Chebyshev polynomial")
    plt.ylabel("Max error")
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    plt.title("Errors Decay")
    plt.grid()
    plt.show()
    return


def solve_BVP1(N, zmin, zmax, L, g, b, bp, x_evaluate):
    scale = 0.5 * (zmax - zmin)
    shift = 0.5 * (zmax + zmin)
    D, x = cheb(N)

    # Right Hand Side
    y = scale*x + shift
    RHS = g(y, scale) - b(x, zmin, zmax, scale) - bp(x, zmin, zmax, scale)

    # enforce boundary condition, u(-1) and u(1) fulfilled by slicing the matrix
    # u'(1) = 0
    L[1, :] = D[0, :]
    RHS[1] = 0

    # u'(-1) = 0
    L[-2, :] = D[-1, :]
    RHS[-2] = 0

    u = np.zeros((N + 1,))
    u[1:N] = np.linalg.solve(L[1:N, 1:N], RHS[1:N])
    q = b(x, zmin, zmax, scale) + u

    # get Chebyshev coefficients
    cheb_coeff = np.zeros((N + 1,))
    t = (np.arange(N + 1)) * np.pi / N
    for k in range(N + 1):
        cheb_coeff[k] = 2 * np.sum(q*np.cos(k*t))/N - q[0]*np.cos(k*t[0])/N - q[N]*np.cos(k*t[-1])/N

    # evaluate q at x_evaluate
    N_eval = np.size(x_evaluate)
    q_out = np.zeros_like(x_evaluate)
    for j in range(N_eval):
        q_out[j] = ChebSum2_Clenshaw_matrix(N, x_evaluate[j], cheb_coeff)
    return q_out


def cheb(N):
    if N == 0:
        D = 0
        x = 1
        return D, x
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones((N + 1,))
    c[0] = 2
    c[N] = 2
    c[1::2] = -c[1::2]

    # create an (N+1)-by-(N+1) matrix whose columns are the vectors of Chebychev nodes
    X = np.zeros((N + 1, N + 1))
    for j in range(N + 1):
        X[:, j] = x
    dX = X - np.transpose(X)
    D = np.outer(c, 1 / c) / (dX + np.eye(N + 1))  # off-diagonal entries
    D = D - np.diag(np.sum(D, axis=1))  # D[i,i] = -\sum_{j\neq i} D[i,j]
    return D, x


def ChebSum2_Clenshaw_matrix(n, x, c):
    diagonals = [np.ones((n + 1,)), -2 * x * np.ones((n,)), np.ones((n - 1,))]
    A = scipy.sparse.diags(diagonals, [0, -1, -2], shape=(n + 1, n + 1))
    A = A.toarray()
    b = scipy.linalg.solve_triangular(np.transpose(A), c, lower=False)  # Upper-triangular solver
    return 0.5 * (b[0] - b[2] - b[n] * np.cos(n * np.arccos(x)))


if __name__ == '__main__':
    main()
