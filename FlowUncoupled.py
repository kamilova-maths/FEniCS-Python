from dolfin import *

# Define mesh and geometry
mesh = RectangleMesh(Point(-1, 0), Point(1, 1), 120, 120)
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


# mu = Constant(0.01)
mu = Expression('exp(-a*pow(x[0],2))', degree=2, a=10)

u_in = Constant(-2.0)
u_c = Constant(-1.0)

# Note, x[0] is r and x[1] is x, and x[1] == 0 is the bottom. 
inflow = 'near(x[1], 1.0) && x[0]<0.5 && x[0]>-0.5'
wall1 = 'near(x[0], 1.0)'
wall2 = 'near(x[0], -1.0)'
# centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'
bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
bcu_wall1 = DirichletBC(W.sub(0), (0.0, u_c), wall1)
bcu_wall2 = DirichletBC(W.sub(0), (0.0, u_c), wall2)
bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
# bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcs = [bcu_inflow, bcu_wall1, bcu_wall2, bcu_outflow]
# Define stress tensor
# epsilon = sym(grad(u))
x = SpatialCoordinate(mesh)


def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


# Define the variational form
f = Constant((0, -1))

colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
colors.set_all(0)  # default to zero
# We match the colours to the defined sketch in the Fenics chapter
# CompiledSubDomain("x[0] == 0").mark(colors, 5)
CompiledSubDomain("near(x[1], 1.0) && x[0]<0.5 && x[0]>-0.5 ").mark(colors, 1)
# CompiledSubDomain("near(x[1] == 1) && x[0]>-0.5 ").mark(colors, 1)
CompiledSubDomain("near(x[1], 1.0) && x[0]>=0.5").mark(colors, 2)
# CompiledSubDomain("near(x[1], 1.0) && x[0]>=0.5").mark(colors, 2)
# CompiledSubDomain("near(x[0], 1.0) || near(x[0], -1.0)").mark(colors, 3) # walls
CompiledSubDomain("near(x[0], 1.0)").mark(colors, 3)  # walls
# CompiledSubDomain("x[0] == 1").mark(colors, 3)# wall
CompiledSubDomain("near(x[1], 0.0)").mark(colors, 4)  # outflow
CompiledSubDomain("near(x[0], -1.0)").mark(colors, 6)  # wall2
CompiledSubDomain("near(x[1], 1.0) && x[0]<=-0.5").mark(colors, 7)
# Create the measure
ds = Measure("ds", subdomain_data=colors)

I2 = Identity(2)
F = (2 * mu * inner(epsilon(u), grad(v)) - div(u) * q - div(v) * p - dot(f, v)) * dx + (dot(dot(p * I2, v), n)
                                                                                        - 2 * mu * dot(
            dot(epsilon(u), v),
            n)) * ds(1) + \
    (dot(dot(p * I2, v), n) - 2 * mu * dot(dot(epsilon(u), v), n)) * ds(3) + \
    (dot(dot(p * I2, v), n) - 2 * mu * dot(dot(epsilon(u), v), n)) * ds(4) + \
    (dot(dot(p * I2, v), n) - 2 * mu * dot(dot(epsilon(u), v), n)) * ds(6)

# Solve problem 
solve(F == 0, w, bcs)

# Plot solutions
(u, p) = w.split()
File("StokesWithBC/velocityFull.pvd") << u
File("StokesWithBC/pressureFull.pvd") << p
