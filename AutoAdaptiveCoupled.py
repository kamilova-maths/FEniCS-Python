from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
#domain = Polygon([Point(0.2, 0), Point(0.2, 1), Point(0.1, 1), Point(0, 1), Point(0, 0)])
#mesh = generate_mesh(domain, 8)
mesh = RectangleMesh(Point(0, 0), Point(0.2, 1), 40, 40, "crossed")
#mesh = Mesh('Meshes/CoupledRefinedMeshGamma5Cartesian.xml')
n = FacetNormal(mesh)

# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2)
Q = FiniteElement("CG", triangle, 1)
S = FiniteElement("CG", triangle, 1)
W = FunctionSpace(mesh, MixedElement([V, Q, S]))

# Define Function and TestFunction(s)
w = Function(W)
(u, p, T) = split(w)
(v, q, s1) = split(TestFunction(W))

# Define the viscosity and bcs
# Qfun should be 2.3710
Qc2 = Expression("Qfun*exp ( -pow( x[1] -(( x1-x2 )/2 + x2), 2 )/( 2*pow( x1-x2,2 ) ) )", degree=3, Qfun=2.3710, x1=0.3,
                x2=0.1)
Qc2.Qfun = 2*2.3710
Ta = Expression("1-x[1]", degree=1)

Gamma = Constant(5.0)
Pe = Constant(27.0)
Bi = Constant(11.6)
# Cartesian
u_in = Constant(-2.0)

# Cylindric
u_in_cyl = Constant(-4.0)

u_c = Constant(-1.0)

inflow = 'near(x[1], 1.0) && x[0]<=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'
bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
bcu_inflow_cyl = DirichletBC(W.sub(0), (0.0, u_in_cyl), inflow)
bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcT_inflow = DirichletBC(W.sub(2), 0.0, inflow)
bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_symmetry, bcT_inflow]
bcs_cyl = [bcu_inflow_cyl, bcu_wall, bcu_outflow, bcu_symmetry, bcT_inflow]

x = SpatialCoordinate(mesh)


def mu(T_local=None):
    if T_local==None:
        T_local = T
    return exp(-Gamma*T_local)


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
    return 2*mu()*epsilon(v)-Id(p)


# stress tensor
def sigma_cyl(v, p):
    return 2*mu()*epsilon_cyl(v)-Id(p)


def Id(p):
    return as_tensor([[p, 0, 0],
                      [0, p, 0],
                     [0, 0, p]])


def cond(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


def sigmabc(v, p):
    return 2*mu()*cond(v) - p*Identity(2)


def div_cyl(v):
    return (1/x[0])*(x[0]*v[0]).dx(0) + v[1].dx(1)


# Define the variational form
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

# Cartesian
a = (inner(sigma(u, p), epsilon(v)) - div(u) * q + (dot(u, grad(T)) * s1 + (
       1 / Pe) * inner(grad(s1), grad(T)))) * dx() - (1 / Pe) * (-Bi * s1 * T * ds(2))
L = (- dot(f, v) - Qc2*s1) * dx() - (1/Pe) * (s1 * Bi * Ta * ds(2))


# Cylindric
a_cyl = (inner(sigma_cyl(u, p), epsilon_cyl(v)) - div_cyl(u) * q + (dot(u, grad(T)) * s1 + (
       1 / Pe) * inner(grad(s1), grad(T)))) * x[0] * dx() - (1 / Pe) * (-Bi * s1 * T * x[0] * ds(2))
L_cyl = (- dot(f, v) - Qc2*s1) * x[0] * dx() - (1/Pe) * (s1 * Bi * Ta * x[0] * ds(2))

F = a + L

F_cyl = a_cyl + L_cyl
# Define error tolerance
tol = 1.e-7

dw = TrialFunction(W)
J = derivative(F, w, dw)
J_cyl = derivative(F_cyl, w, dw)
M = inner(w[0], w[0]) * dx()
M_cyl = inner(w[0], w[0]) * x[0] * dx()
#problem = NonlinearVariationalProblem(F, w, bcs, J)
problem_cyl = NonlinearVariationalProblem(F_cyl, w, bcs_cyl, J_cyl)
#solver = AdaptiveNonlinearVariationalSolver(problem, M)
solver = AdaptiveNonlinearVariationalSolver(problem_cyl, M_cyl)
#solver = NonlinearVariationalSolver(problem)
solver.parameters["error_control"]["dual_variational_solver"]["linear_solver"] = "umfpack"
#solver.parameters["nonlinear_solver"] = "cg"
#solver.parameters["krylov_solver"]["preconditioner"] = "amg"
#solver.parameters["newton_solver"]["linear_solver"] = "umfpack"
parameters["refinement_algorithm"] = "plaza_with_parent_facets"
solver.solve(tol)
#solver.solve()
# Extract solution
solver.summary()

#Plot solutions
(u0, p0, T0) = w.root_node().split()
(u1, p1, T1) = w.leaf_node().split()

R = w.leaf_node().function_space()
plot(R.mesh())
plt.show()


plot(u0, title="Velocity on initial mesh")
plt.show()
plot(u1, title="Velocity on final mesh")
plt.show()

plot(p0, title="Pressure on initial mesh")
plt.show()
plot(p1, title="Pressure on final mesh")
plt.show()

plot(T0, title="Temperature on initial mesh")
plt.show()
plot(T1, title="Temperature on final mesh")
plt.show()


#Saving data
# File("Results/velocityDecoupledInitialMesh.pvd") << u0
# File("Results/velocityDecoupledFinalMesh.pvd") << u1
# File("Results/pressureDecoupledInitialMesh.pvd") << p0
# File("Results/pressureDecoupledFinalMesh.pvd") << p1
# File("Results/TemperatureDecoupledInitialMesh.pvd") << T0
# File("Results/TemperatureDecoupledFinalMesh.pvd") << T1
File("Meshes/CoupledRefinedMeshQTwiceGamma5Cyl.xml") << R.mesh()
