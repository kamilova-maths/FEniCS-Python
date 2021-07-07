from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
# THIS DOESN'T WORK, DON'T WASTE YOUR TIME OKAY.
mesh = Mesh('Meshes/CoupledRefinedMeshGamma5Cartesian.xml')
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

Ta = Expression("1-x[1]", degree=1)

Gamma = Constant(23.0)
Pe = Constant(27)
Bi = Constant(11.6)
#Bi = Constant(0.0)
#Bi = Constant(200.0)
val = "Bi200" # whatever I am testing at the time
# Cartesian
u_in = Constant(-2.0)

# Cylindric
u_in_cyl = Constant(-4.0)

u_c = Constant(-1.0)

inflow = 'near(x[1], 1.0) && x[0]<=0.1'
right = 'near(x[1], 1.0) && x[0] >=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'
bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
bcu_inflow_cyl = DirichletBC(W.sub(0), (0.0, u_in_cyl), inflow)
bcu_right = DirichletBC(W.sub(0).sub(1), 0.0, right)
bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
bcP_outflow = DirichletBC(W.sub(1), 0.0, outflow)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcT_inflow = DirichletBC(W.sub(2), 0.0, inflow)
bcs = [bcu_inflow, bcu_wall, bcP_outflow, bcu_symmetry, bcu_right, bcT_inflow]
bcs_cyl = [bcu_inflow_cyl, bcu_wall, bcP_outflow, bcu_symmetry, bcT_inflow]
x = SpatialCoordinate(mesh)


def mu(T_local=None):
    if T_local==None:
        T_local = T
    return exp(-Gamma*T_local)


def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


# symmetric cylindric gradient
def epsilon_cyl(v):
    return sym(as_tensor([[v[0].dx(0), 0, v[0].dx(1)],
                          [0, v[0] / x[0], 0],
                          [v[1].dx(0), 0, v[1].dx(1)]]))


# stress tensor
def sigma(v, p):
    return 2*mu()*epsilon(v)-p*Identity(2)


# stress tensor
def sigma_cyl(v, p):
    return 2*mu()*epsilon_cyl(v)-Id(p)


def Id(p):
    return as_tensor([[p, 0, 0],
                      [0, p, 0],
                     [0, 0, p]])




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

# Create the measure
ds = Measure("ds", domain=mesh, subdomain_data=facet_f)
coord = "Cyl" # can also be "Cyl"

ind = 0 # 0 is Cartesian, 1 is Cylindric. Let's not make it too complicated, definitely fix it later.
dw = TrialFunction(W)
if ind == 1:
# Cartesian
    a1 = inner(sigma(u, p), epsilon(v)) * dx() - dot(f, v) * dx()
    a = (inner(sigma(u, p), epsilon(v)) - div(u) * q + (dot(u, grad(T)) * s1 + (
           1 / Pe) * inner(grad(s1), grad(T)))) * dx() - (1 / Pe) * (-Bi * s1 * T * ds(2)) - s1*dot(grad(T), n)*ds(0)
    b0 = - dot(dot(sigma(u, p), v), n) * ds(0)
    b2 = - dot(dot(sigma(u, p), v), n) * ds(2)
    b3 = -dot(dot(sigma(u, p), v), n) * ds(3)
    b4 = + dot(p*n, v) * ds(4)
    #b4 = -dot(dot(sigma2d(u, p), v), n) * ds(4)
    b = b0 + b2 + b3 + b4
    L = (- dot(f, v) - Qc2*s1) * dx() - (1/Pe) * (s1 * Bi * Ta * ds(2))

    F = a + L + b
    J = derivative(F, w, dw)
else:
    # Cylindric
    print("I am cylindric")
    a_cyl = (inner(sigma_cyl(u, p), epsilon_cyl(v)) - div_cyl(u) * q + (dot(u, grad(T)) * s1 + (
           1 / Pe) * inner(grad(s1), grad(T)))) * x[0] * dx() - (1 / Pe) * (-Bi * s1 * T * x[0] * ds(2))
    L_cyl = (- dot(f, v) - Qc2*s1) * x[0] * dx() - (1/Pe) * (s1 * Bi * Ta * x[0] * ds(2)) - s1*dot(grad(T), n)* x[0] * ds(0)
    b0_cyl = - dot(dot(sigma(u, p), v), n) * x[0] * ds(0)
    b2_cyl = - dot(dot(sigma(u, p), v), n) * x[0] * ds(2)
    b3_cyl = -dot(dot(sigma(u, p), v), n) * x[0] * ds(3)
    b4_cyl = + dot(p*n, v) * x[0] * ds(4)
    b_cyl = b0_cyl + b2_cyl + b3_cyl + b4_cyl

    F_cyl = a_cyl + L_cyl + b_cyl
    F = F_cyl
    J_cyl = derivative(F_cyl, w, dw)
    J = J_cyl
    bcs = bcs_cyl



print("The coordinate system is ", coord)
for Gamma_val in [0.1, 0.5, 1, 5, 10, 15, 20, 23]:
    Gamma.assign(Gamma_val)
    print('Gamma =', Gamma_val)
    problem = NonlinearVariationalProblem(F, w, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["linear_solver"] = 'umfpack'

    solver.solve()
    # Extract solution
    (u, p, T) = w.split()
#solver.summary()

Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
sig = Function(Vsig, name="Stress" + coord)
sig.assign(project(sigma(u, p), Vsig))

normal_stress0 = assemble(inner(sig*n, n)*ds(0))
flux1 = assemble(dot(u, n) * dot(u, n) * ds(1))
values = np.asarray([normal_stress0, flux1])
print("Flux 1", flux1)

#
sav = 1.0
if sav == 1.0:
# Saving data
    np.savetxt("Results/valuesCoupledSlip" + coord  + "BiOrgAdaptiveMesh.csv", values.T, delimiter='\t')
    File("Results/velocityCoupledSlip" + coord  + "BiOrgAdaptiveMesh.pvd") << u
    File("Results/pressureCoupledSlip" + coord + "BiOrgAdaptiveMesh.pvd") << p
    File("Results/TemperatureCoupledSlip" + coord + "BiOrgAdaptiveMesh.pvd") << T
    File("Results/StressCoupledSlip" + coord + "BiOrgAdaptiveMesh.pvd") << sig
    W2 = FunctionSpace(mesh, S)
    Pmu = project(mu(), W2)

    File("Results/ViscosityCoupled" + coord + "BiOrgAdaptedMesh.pvd") << Pmu

