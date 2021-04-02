from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

# Choose a mesh. You can also import a previously adapted mesh

#domain = Polygon([Point(0.2, 0), Point(0.2, 1), Point(0.1, 1), Point(0, 1), Point(0, 0)])
#mesh = generate_mesh(domain, 8)
#mesh = RectangleMesh(Point(0, 0), Point(0.2, 1), 70, 70)
#mesh = Mesh('Meshes/CoupledRefinedMeshGamma5Cartesian.xml')

# Define mesh and geometry
def mesh_ref(lx,ly, Nx,Ny):
    m = UnitSquareMesh(Nx, Ny)
    x = m.coordinates()

    #Refine on top and bottom sides
    x[:,1] = (x[:,1] - 0.5) * 2.
    x[:,1] = 0.5 * (np.cos(pi * (x[:,1] - 1.) / 2.) + 1.)
    # x[:, 0] = (x[:, 0] - 0.5) * 2.
    # x[:, 0] = 0.5 * (np.cos(pi * (x[:, 0] - 1.) / 2.) + 1.)
    #Scale
    x[:,0] = x[:,0]*lx
    x[:,1] = x[:,1]*ly

    return m

mesh = mesh_ref(0.2, 1.0, 30, 30)

#mesh = refine(mesh)

plot(mesh)
plt.show()

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
bcP_outflow = DirichletBC(W.sub(1), 0.0, outflow)
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


# these should be the same for cylindric and Cartesian... I think ... x.x
def epsilon2d(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


def sigma2d(v, p):
    return 2*mu()*epsilon2d(v) - p*Identity(2)


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

# Cartesian
a = (inner(sigma(u, p), epsilon(v)) - div(u) * q + (dot(u, grad(T)) * s1 + (
       1 / Pe) * inner(grad(s1), grad(T)))) * dx() - (1 / Pe) * (-Bi * s1 * T * ds(2))
b0 = - dot(dot(sigma2d(u, p), v), n) * ds(0) - s1*dot(grad(T), n)*ds(0)
b2 = - dot(dot(sigma2d(u, p), v), n) * ds(2)
b3 = -dot(dot(sigma2d(u, p), v), n) * ds(3)
b4 = + dot(p*n, v) * x[0] * ds(4)
L = (- dot(f, v) - Qc2*s1) * dx() - (1/Pe) * (s1 * Bi * Ta * ds(2))


# Cylindric
a_cyl = (inner(sigma_cyl(u, p), epsilon_cyl(v)) - div_cyl(u) * q + (dot(u, grad(T)) * s1 + (
       1 / Pe) * inner(grad(s1), grad(T)))) * x[0] * dx() - (1 / Pe) * (-Bi * s1 * T * x[0] * ds(2))
L_cyl = (- dot(f, v) - Qc2*s1) * x[0] * dx() - (1/Pe) * (s1 * Bi * Ta * x[0] * ds(2))

F = a + L + b0 + b2 + b3 + b4

F_cyl = a_cyl + L_cyl

dw = TrialFunction(W)
J = derivative(F, w, dw)
J_cyl = derivative(F_cyl, w, dw)

for Gamma_val in [10, 15, 20, 23]:
    Gamma.assign(Gamma_val)
    print('Gamma =', Gamma_val)
    # Choose either Cylindrical or Cartesian coordinates. Uncomment the relevant one.
    problem = NonlinearVariationalProblem(F, w, bcs, J)
    #problem = NonlinearVariationalProblem(F_cyl, w, bcs_cyl, J_cyl)

    solver = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["linear_solver"] = 'umfpack'

    solver.solve()
    # Extract solution
    (u, p, T) = w.split()
#solver.summary()

Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
sig = Function(Vsig, name="Stress")
sig.assign(project(sigma2d(u, p), Vsig))
area0 = assemble(1.0*ds(0))
area1 = assemble(1.0 * ds(1))
area2 = assemble(1.0 * ds(2))
area3 = assemble(1.0 * ds(3))
area4 = assemble(1.0 * ds(4))
print("area at ds0 is", area0)
print("area at ds1 is", area1)
print("area at ds2 is", area2)
print("area at ds3 is", area3)
print("area at ds4 is", area4)
normal_stress0 = assemble(inner(sig*n, n)*ds(0))/area0
normal_stress1 = assemble(inner(sig * n, n) * ds(1))/area1
normal_stress2 = assemble(inner(sig * n, n) * ds(2))/area2
normal_stress3 = assemble(inner(sig * n, n) * ds(3))/area3
normal_stress4 = assemble(inner(sig * n, n) * ds(4))/area4
print("Stress at (0.1, 1):", sig(0.1, 1))
print("Normal stress at boundary 0", normal_stress0)
print("Normal stress at boundary 1", normal_stress1)
print("Normal stress at boundary 2", normal_stress2)
print("Normal stress at boundary 3", normal_stress3)
print("Normal stress at boundary 4", normal_stress4)

sav = 0.0
if sav == 1.0:
# Saving data
    File("Results/velocityCoupledCylAdaptedMesh.pvd") << u
    File("Results/pressureCoupledCylAdaptedMesh.pvd") << p
    File("Results/TemperatureCoupledCylAdaptedMesh.pvd") << T

    W2 = FunctionSpace(mesh, S)
    Pmu = project(mu(), W2)

    File("Results/ViscosityCoupledCylAdaptedMesh.pvd") << Pmu