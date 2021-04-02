from dolfin import *
from mshr import *
import matplotlib.pyplot as plt

domain = Polygon([Point(0.2, 0), Point(0.2, 1), Point(0.1, 1), Point(0, 1), Point(0, 0)])
mesh = generate_mesh(domain, 60)
n = FacetNormal(mesh)

# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2)
Q = FiniteElement("CG", triangle, 1)
W = FunctionSpace(mesh, MixedElement([V, Q]))

# Define Function and TestFunction(s)
w = TrialFunction(W)
(u, p) = split(w)
(v, q) = split(TestFunction(W))

# Define the viscosity and bcs

mu = Constant(1.0)

# Cartesian velocity that gives mass balance
u_in = Constant(-2.0)

# Cylindric velocity that gives mass balance
u_incyl = Constant(-4.0)

# Wall velocity
u_c = Constant(-1.0)

inflow = 'near(x[1], 1.0) && x[0]<=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'
bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
bcu_inflow_cyl = DirichletBC(W.sub(0), (0.0, u_incyl), inflow)
bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_symmetry]
bcs_cyl = [bcu_inflow_cyl, bcu_wall, bcu_outflow, bcu_symmetry]
x = SpatialCoordinate(mesh)


def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1), 0],
                          [v[1].dx(0), v[1].dx(1), 0],
                          [0, 0, 0]]))


# symmetric cylindric gradient
def epsilon_cyl(v):
    return sym(as_tensor([[v[0].dx(0), 0, v[0].dx(1)],
                          [0, v[0] / x[0], 0],
                          [v[1].dx(0), 0, v[1].dx(1)]]))



# stress tensor
def sigma(v, p):
    return 2*mu*epsilon(v)-Id(p)


# stress tensor
def sigma_cyl(v, p):
    return 2*mu*epsilon_cyl(v)-Id(p)


def Id(p):
    return as_tensor([[p, 0, 0],
                      [0, p, 0],
                     [0, 0, p]])


def cond(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


def sigmabc(v, p):
    return 2*mu*cond(v) - p*Identity(2)


def div_cyl(v):
    return (1/x[0])*(x[0]*v[0]).dx(0) + v[1].dx(1)


f = Constant((0, -1))

facet_f = MeshFunction("size_t", mesh, mesh.topology().dim() - 1) # FACET function
CompiledSubDomain('near(x[1], 1.0) && x[0]<=0.1').mark(facet_f, 0)
CompiledSubDomain('near(x[1], 1.0) && x[0]>=0.1').mark(facet_f, 1)
CompiledSubDomain('near(x[0], 0.2)').mark(facet_f, 2)  # wall
CompiledSubDomain('near(x[1], 0.0)').mark(facet_f, 3)  # outflow
CompiledSubDomain('near(x[0], 0.0)').mark(facet_f, 4)
CompiledSubDomain('near(x[1], 1.0) && x[0]>0.1 && x[0]<0.2').mark(facet_f, 5)
# Create the measure
ds = Measure("ds", domain=mesh, subdomain_data=facet_f)

# Cartesian construction
a = (inner(sigma(u, p), epsilon(v)) - div(u) * q) * dx()
L = (- dot(f, v)) * dx()

# Cylindric construction (multiplied by r )
a_cyl = (inner(sigma_cyl(u, p), epsilon_cyl(v)) - div_cyl(u)*q) * x[0] * dx()
L_cyl = (- dot(f, v)) * x[0] * dx()

# Define function for the solution
w = Function(W)

# Solve equation a = L with respect to u and the given boundary conditions
problem = LinearVariationalProblem(a, L, w, bcs)
problem_cyl = LinearVariationalProblem(a_cyl, L_cyl, w, bcs_cyl)

solver = LinearVariationalSolver(problem_cyl)
#solver = LinearVariationalSolver(problem)

solver.parameters["linear_solver"] = "umfpack"
solver.solve()
(u, p) = w.split()

c = plot(u, title="Velocity")
plt.colorbar(c)
plt.show()
