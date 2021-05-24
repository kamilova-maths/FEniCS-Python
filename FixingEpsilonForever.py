from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

# Define mesh and geometry - We solve for half of the domain we need, and impose symmetry

#mesh = RectangleMesh(Point(0, 0), Point(0.2, 1), 100, 100, "crossed")
mesh = RectangleMesh(Point(0, 0), Point(1, 1), 100, 100, "crossed")
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
#mu = Expression('exp(-a*pow(x[0],2))', degree=2, a=10)

#u_in = Constant(-1.5)
p0 = Constant(-9.0)
p1 = Constant(0.0)
L = 3.0
R = 0.5
#asp = R/L
asp = Constant(0.2)
u_e = Expression(('0', '-4*(x[0]+1)*(1-x[0])/(asp)'), asp=0.2, degree=2)

# Note, x[0] is r and x[1] is x, and x[1] == 0 is the bottom.
inflow = 'near(x[1], 1.0)'
#inflow = 'near(x[1], 5.0)'
wall1 = 'near(x[0], 1.0)'
#wall1 = 'near(x[0], 0.2)'
wall2 = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'
bcu_inflow = DirichletBC(W.sub(0), u_e,  inflow)
bcu_wall1 = DirichletBC(W.sub(0), u_e, wall1)
bcu_wall2 = DirichletBC(W.sub(0), u_e, wall2)
bcu_outflow = DirichletBC(W.sub(1), p1, outflow)
bcs = [bcu_inflow, bcu_wall1, bcu_wall2, bcu_outflow]
# Define stress tensor
# epsilon = sym(grad(u))
x = SpatialCoordinate(mesh)


def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


# stress tensor
def sigma(v, p):
    return 2*mu*epsilon(v)-p*Identity(2)




# Define the variational form
f = Constant((0, -1))


vsc = as_vector([v[0], asp*v[1]])
usc = as_vector([u[0], asp*u[1]])
colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
colors.set_all(0)  # default to zero
# We match the colours to the defined sketch in the Fenics chapter
CompiledSubDomain("near(x[0], 0.0)").mark(colors, 5)
CompiledSubDomain("near(x[1], 1.0)").mark(colors, 1)
#CompiledSubDomain("near(x[1], 5.0)").mark(colors, 1)
CompiledSubDomain("near(x[0], 1.0)").mark(colors, 3)  # wall
#CompiledSubDomain("near(x[0], 0.2)").mark(colors, 3)  # wall
CompiledSubDomain("near(x[1], 0.0)").mark(colors, 4)  # outflow

# Create the measure
ds = Measure("ds", subdomain_data=colors)

a1 = (inner(sigma(usc, p), epsilon(vsc))) * dx
a2 = (- div(usc) * q - dot(f, vsc)) * dx
b1 = - dot(dot(sigma(usc, p), vsc), n) * ds
#b3 = - dot(dot(sigmabc(usc, p), vsc), n) * ds(3)
#b4 = - asp*dot(dot(sigmabc(usc, p), vsc), n) * ds(4)
F = a1 + a2 + b1

# Solve problem
solve(F == 0, w, bcs)

# Plot solutions
(u, p) = w.split()
plt.figure()
c = plot(p, title='pressure')
plt.colorbar(c)
plt.show()
# Compute error

W2 = FunctionSpace(mesh, V)
u_e = interpolate(u_e, W2)
u_r = interpolate(u, W2)
error = np.abs(np.array(u_e.vector()) - np.array(u_r.vector())).max()

print('error = ', error)
print('max u:', np.array(u_r.vector()).max())

p_e = Expression('7*x[1]', degree=1)
W3 = FunctionSpace(mesh, Q)

vertex_values_p_e = p_e.compute_vertex_values(mesh)
vertex_values_p_r = p.compute_vertex_values(mesh)

error = np.max(np.abs(vertex_values_p_e - vertex_values_p_r))
print('error pressure = ', error)
p_e = interpolate(p_e, W3)
cond = "YesEps"

File("Results/velocity" + cond + ".pvd") << u
File("Results/pressure" + cond + ".pvd") << p
File("Results/velocityAna" + cond + ".pvd") << u_e
File("Results/pressureAna" + cond + ".pvd") << p_e

