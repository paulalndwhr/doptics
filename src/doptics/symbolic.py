from sympy import Function, symbols, solve, lambdify, Eq, diff


def u_prime_mv():
    x, y = symbols("x,y")
    L1, L2 = symbols("L1, L2")
    V = Function("V")(y)
    u = Function("u")(x)
    w = Function("w")(y)
    t1 = Function("t1")(y)
    t2 = Function("t2")(y)

    expr = V**2 - 2*u*V-2*w*V+2*u*w-y**2-x**2+2*x*y+2*y*w*t1-2*x*w*t1-L1**2+2*L1*w*t2+2*L1*u-2*w*t2*u

    eq = Eq(expr, 0)

    sol = solve(eq, w)[0]

    dudx = solve(Eq(diff(sol, x), 0), diff(u))[0].simplify()

    return lambdify([x, y, u, V, t1, t2, L1], dudx), lambdify([x, y, u, V, t1, t2, L1], sol)
