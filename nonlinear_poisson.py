""" Solve a nonlinear variational problem with Firedrake. """
import firedrake as fe


mesh = fe.UnitIntervalMesh(4)

element = fe.FiniteElement("P", mesh.ufl_cell(), 1)

V = fe.FunctionSpace(mesh, element)

u = fe.Function(V)

v = fe.TestFunction(V)

bc = fe.DirichletBC(V, 0., "on_boundary")

alpha = 1 + u/10.

x = fe.SpatialCoordinate(mesh)[0]

div, grad, dot, sin, pi, dx = fe.div, fe.grad, fe.dot, fe.sin, fe.pi, fe.dx

f = 10.*sin(pi*x)

R = (-dot(grad(v), alpha*grad(u)) - v*f)*dx


problem = fe.NonlinearVariationalProblem(R, u, bc, fe.derivative(R, u))

solver = fe.NonlinearVariationalSolver(
    problem, solver_parameters = {"snes_monitor": True})

solver.solve()

print("u_h = " + str(u.vector()[:]))
