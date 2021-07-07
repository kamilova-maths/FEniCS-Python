from fenics import *
from dolfin import *
from dolfin_dg import StokesNitscheBoundary, tangential_proj
from mshr import *
import matplotlib.pyplot as plt
import numpy as np


#mesh = Mesh('Meshes/CoupledRefinedMeshGamma5Cartesian.xml')
mesh = RectangleMesh(Point(0, 0), Point(0.2, 1), 40, 40, "crossed")

mesh = Mesh('Meshes/CoupledRefinedMeshQTwiceGamma5Cyl.xml')
n = FacetNormal(mesh)
x = SpatialCoordinate(mesh)

# Cartesian
u_in = Constant(-2.0)
# Cylindric
u_in_cyl = Constant(-4.0)
u_c = Constant(-1.0)
u_soln = Constant((0.0, 0.0)) # This is what is dotted with the normal component, and it tells me what the normal
    # component of the velocity is. Namely, if this is zero, then I am setting the NORMAL component of the velocity equal to zero.
Gamma = Constant(5.0)


# Note, x[0] is r and x[1] is x, and x[1] == 0 is the bottom.
inflow = 'near(x[1], 1.0) && x[0]<=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'
#weird = 'near(x[1], 1.0) && x[0]>=0.1'


# Viscous flux operator
def F_v(u, grad_u, p_local=None):
    if p_local is None:
        p_local = p
    return 2 * eta() * sym(grad_u) - p_local * Identity(2)


def F_v_cyl(u, grad_u, p_local=None):
    if p_local is None:
        p_local = p
    return 2 * eta() * sym(grad_u) - p_local * Identity(3)


def grad_cyl(v):
    return as_tensor([[v[0].dx(0), 0, v[0].dx(1)],
                          [0, v[0] / x[0], 0],
                          [v[1].dx(0), 0, v[1].dx(1)]])


# Viscosity model
def eta(T_local=None):
    if T_local is None:
        T_local = T
    return exp(-Gamma * T_local)


def div_cyl(v):
    return (1/x[0])*(x[0]*v[0]).dx(0) + v[1].dx(1)


total_stress_old = 0.0
stress_array = []

Ta = Expression("1-x[1]", degree=1)

Pe = Constant(27.0)
Bi = Constant(11.6)
#Bi = Constant(0.0)
Qc = Constant(0.0)

f = Constant((0.0, -1.0))

g_tau = Constant((0.0, 0.0))  # Tangential stress component

# Function space
We = MixedElement([VectorElement("CG", mesh.ufl_cell(), 2), FiniteElement("CG", mesh.ufl_cell(), 1), FiniteElement("CG", mesh.ufl_cell(), 1)])
W = FunctionSpace(mesh, We)
w = Function(W)

bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcT_inflow = DirichletBC(W.sub(2), 0.0, inflow)
bcP_outflow = DirichletBC(W.sub(1), 0.0, outflow)
bcu_inflow_cyl = DirichletBC(W.sub(0), (0.0, u_in_cyl), inflow)
bcs = [bcu_inflow, bcu_wall, bcP_outflow, bcu_symmetry, bcT_inflow]
bcs_cyl = [bcu_inflow_cyl, bcu_wall, bcP_outflow, bcu_symmetry, bcT_inflow]

facet_f = MeshFunction("size_t", mesh, mesh.topology().dim() - 1) # FACET function

CompiledSubDomain('near(x[1], 1.0) && x[0]<=0.1').mark(facet_f, 0)
CompiledSubDomain('near(x[1], 1.0) && x[0]>=0.1').mark(facet_f, 1)
CompiledSubDomain('near(x[0], 0.2)').mark(facet_f, 2)  # wall
CompiledSubDomain('near(x[1], 0.0)').mark(facet_f, 3)  # outflow
CompiledSubDomain('near(x[0], 0.0)').mark(facet_f, 4)

# Create the measure
ds = Measure("ds", domain=mesh, subdomain_data=facet_f)
# Variational formulation
u, p, T = split(w)
v, q, S = split(TestFunction(W))

Qc2 = Expression("Qfun*exp(-pow( x[1] -( x1-x2 )/2, 2 )/( 2*pow( x1-x2,2 ) ) )", degree=1, Qfun=2.3710, x1=0.3, x2=0.1)
#Qc2.Qfun = 0.25
val = "Org"
coord = "Cyl" # can also be "Cyl"

ind = 0 # 0 is Cartesian, 1 is Cylindric. Let's not make it too complicated, definitely fix it later.


if ind==1:
    N = inner(F_v(u, grad(u)), grad(v)) * dx - dot(f, v) * dx - div(u) * q * dx + dot(u, grad(T))*S*dx +\
        (1/Pe) * inner(grad(S), grad(T)) * dx - Qc2*S*dx
    u_bt = - dot(dot(F_v(u, grad(u), p), v), n)*ds(3) - dot(dot(F_v(u, grad(u), p), v), n)*ds(2) \
        - dot(dot(F_v(u, grad(u), p), v), n)*ds(0)
    T_bt = - (1/Pe) * ( - Bi * S * T * ds(2) + S * Bi * Ta * ds(2)) - S*dot(grad(T), n)* x[0] * ds(0)
    N += T_bt
    N += u_bt
    # Slip boundary conditions
    stokes_nitsche = StokesNitscheBoundary(F_v, u, p, v, q)
    N += stokes_nitsche.slip_nitsche_bc_residual(u_soln, g_tau, ds(1))
    #solve(N == 0, w, bcs, solver_parameters={"newton_solver":{"relative_tolerance":1e-9}})

    # Qc2.Qfun = 2.3710
    #
    # solve(N == 0, w, bcs, solver_parameters={"newton_solver":{"relative_tolerance":1e-9}})

else:
    N = inner(F_v_cyl(u, grad_cyl(u)), grad_cyl(v)) * x[0] * dx - dot(f, v) * x[0] * dx - div_cyl(u) * q * x[0] * dx + dot(u, grad(T)) * S * x[0] * dx + \
        (1 / Pe) * inner(grad(S), grad(T)) * x[0] * dx - Qc2 * S * x[0] * dx
    u_bt = - dot(dot(F_v(u, grad(u), p), v), n) * x[0]* ds(3) - dot(dot(F_v(u, grad(u), p), v), n) * x[0] * ds(2) \
           - dot(dot(F_v(u, grad(u), p), v), n) * x[0] * ds(0)
    T_bt = - (1 / Pe) * (- Bi * S * T * x[0] * ds(2) + S * Bi * Ta * x[0] * ds(2))
    N += T_bt
    N += u_bt
    # Slip boundary conditions
    stokes_nitsche = StokesNitscheBoundary(F_v, u, p, v, q)
    N += stokes_nitsche.slip_nitsche_bc_residual(u_soln, g_tau, ds(1))
    # solve(N == 0, w, bcs_cyl, solver_parameters={"newton_solver": {"relative_tolerance": 1e-9}})
    # Qc2.Qfun = 2.3710
    # solve(N == 0, w, bcs_cyl, solver_parameters={"newton_solver": {"relative_tolerance": 1e-9}})
    bcs = bcs_cyl
    # Plot solutions

dw = TrialFunction(W)
J_cyl = derivative(N, w, dw)

M_cyl = inner(w[0], w[0]) * x[0] * dx()
tol = 1e-7
problem_cyl = NonlinearVariationalProblem(N, w, bcs, J_cyl)
solver = AdaptiveNonlinearVariationalSolver(problem_cyl, M_cyl)
solver.parameters["error_control"]["dual_variational_solver"]["linear_solver"] = "umfpack"
parameters["refinement_algorithm"] = "plaza_with_parent_facets"
solver.solve(tol)

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
File("Meshes/CoupledRefinedMeshSlip.xml") << R.mesh()
