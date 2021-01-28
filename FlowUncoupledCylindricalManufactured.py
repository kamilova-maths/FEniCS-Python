from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

# Define mesh and geometry - We solve for half of the domain we need, and impose symmetry
mesh = RectangleMesh(Point(0, 0), Point(1, 1), 100, 100)
n = FacetNormal(mesh)

# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2)
Q = FiniteElement("CG", triangle, 1)
W = FunctionSpace(mesh, MixedElement([V, Q]))

# Define Function and TestFunction(s)
w = Function(W)
(u, p) = split(w)
(v, q) = split(TestFunction(W))

# Define the viscosity and bcs

mu = Constant(1)

u_c = Constant(0.0)
p1 = Constant(0.0)

# Analytic expression for the velocity
u_in = Expression(('0', '-2*(1-x[0]*x[0])'), degree=2)

# Note, x[0] is r and x[1] is x, and x[1] == 0 is the bottom.
inflow = 'near(x[1], 1.0)'
wall1 = 'near(x[0], 1.0)'
wall2 = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'
bcu_inflow = DirichletBC(W.sub(0), u_in,  inflow)
bcu_wall1 = DirichletBC(W.sub(0), (0.0, 0.0), wall1)
bcu_wall2 = DirichletBC(W.sub(0), (0.0, -2.0), wall2)
bcs = [bcu_inflow, bcu_wall1, bcu_wall2]

x = SpatialCoordinate(mesh)


# symmetric gradient
def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), 0, v[0].dx(1)],
                          [0, v[0] / x[0], 0],
                          [v[1].dx(0), 0, v[1].dx(1)]]))

#gradient
def epsilon2(v):
    return as_tensor([[v[0].dx(0), 0, v[0].dx(1)],
                          [0, v[0] / x[0], 0],
                          [v[1].dx(0), 0, v[1].dx(1)]])


# stress tensor
def sigma(v, p):
    return 2*mu*epsilon(v)-p*Identity(3)


def div_cyl(v):
    return (1/x[0])*(x[0]*v[0]).dx(0) + v[1].dx(1)


# analytic expression of what the stress tensor should be with my analytic velocity
def sigmabc2(v):
    return 2*mu*sym(as_tensor([[0, 0],
                               [4*x[0], 0]]))


# Define the variational form
f = Constant((0, -1))

colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
colors.set_all(0)
CompiledSubDomain("near(x[0], 0.0)").mark(colors, 5)
CompiledSubDomain("near(x[1], 1.0)").mark(colors, 1)
CompiledSubDomain("near(x[0], 1.0)").mark(colors, 3)  # wall
CompiledSubDomain("near(x[1], 0.0)").mark(colors, 4)  # outflow
vext = as_vector([v[0], 0, v[1]])
next = as_vector([n[0], 0, n[1]])
# Create the measure
ds = Measure("ds", subdomain_data=colors)
#ap1 = (2*u[0].dx(0)*v[0].dx(0) + (u[0].dx(1) + u[1].dx(0))*(v[0].dx(1) + v[1].dx(0)) + 2*(u[0] / x[0]) * (v[0] / x[0]) + 2*u[1].dx(1)*v[1].dx(1)) * x[0] * dx
#ap2 = (-p*v[0].dx(0) -p*(v[0] / x[0]) - p*v[1].dx(1)) * x[0] * dx
#a1 = ap1 + ap2

# Here I can use either the symmetric gradient or the non-symmetric gradient.
a1 = (inner(sigma(u, p), epsilon(v))) * x[0] * dx
a2 = (- div_cyl(u) * q - dot(f, v)) * x[0] * dx
#a2 = (- ((1/x[0])*(x[0]*u[0]).dx(0) + u[1].dx(1)) * q - f[1]*v[1]) * x[0] * dx
b1 = - dot(dot(sigmabc2(u), v), n) * x[0] * ds(4)


F = a1 + a2 + b1

# Solve problem
solve(F == 0, w, bcs)

# Plot solutions
(u, p) = w.split()
File("StokesWithBC/pressureCyl.pvd") << p
plt.figure()
c = plot(p, title='pressure')
plt.colorbar(c)
plt.show()
# Compute error
u_e = Expression(('0', '-2*(1-x[0]*x[0])'), degree=2)
W2 = FunctionSpace(mesh, V)
u_e = interpolate(u_e, W2)
u_r = interpolate(u, W2)
error = np.abs(np.array(u_e.vector()) - np.array(u_r.vector())).max()

print('error velocity = ', error)
print('max u:', np.array(u_r.vector()).max())

p_e = Expression('7*x[1]', degree=1)
W3 = FunctionSpace(mesh, Q)

vertex_values_p_e = p_e.compute_vertex_values(mesh)
vertex_values_p_r = p.compute_vertex_values(mesh)

error = np.max(np.abs(vertex_values_p_e - vertex_values_p_r))
print('error pressure = ', error)

File("StokesWithBC/velocityCyl.pvd") << u
File("StokesWithBC/velocityAnalyticCyl.pvd") << u_e
