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

mesh = refine(mesh)


n = FacetNormal(mesh)

# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2)
Q = FiniteElement("CG", triangle, 1)
#S = FiniteElement("CG", triangle, 1)
W = FunctionSpace(mesh, MixedElement([V, Q]))

# Define Function and TestFunction(s)
w = Function(W)
(u, p) = split(w)
(v, q) = split(TestFunction(W))

# Define the viscosity and bcs
# Qfun should be 2.3710
#Qc2 = Expression("Qfun*exp ( -pow( x[1] -(( x1-x2 )/2 + x2), 2 )/( 2*pow( x1-x2,2 ) ) )", degree=3, Qfun=2.3710, x1=0.3,
 #               x2=0.1)

#Ta = Expression("1-x[1]", degree=1)

Gamma = Constant(23.0)
Pe = Constant(27.0)
Bi = Constant(11.6)
# Cartesian - We must modify the inlet and outlet velocities so that they  match the rescaled values in EVV chapter
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
bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_symmetry]
bcs_cyl = [bcu_inflow_cyl, bcu_wall, bcu_outflow, bcu_symmetry]
x = SpatialCoordinate(mesh)
T = Expression("x[0]*x[0]/(asp*asp)", asp=0.2, degree=2)
Texp = "Quadratic"
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
# a1 = inner(sigma(u, p), epsilon(v)) * dx() - dot(f, v) * dx()
a = (inner(sigma(u, p), epsilon(v)) - div(u) * q) * dx()
b0 = - dot(dot(sigma2d(u, p), v), n) * ds(0)
b2 = - dot(dot(sigma2d(u, p), v), n) * ds(2)
b3 = -dot(dot(sigma2d(u, p), v), n) * ds(3)
b4 = + dot(p*n, v) * ds(4)
#b4 = -dot(dot(sigma2d(u, p), v), n) * ds(4)
b = b0 + b2 + b3 + b4
L = (- dot(f, v)) * dx()
F = a + L + b
# Cylindric
a_cyl = (inner(sigma_cyl(u, p), epsilon_cyl(v)) - div_cyl(u) * q ) * x[0] * dx()
L_cyl = (- dot(f, v) ) * x[0] * dx()
b0_cyl = - dot(dot(sigma2d(u, p), v), n) * x[0] * ds(0)
b2_cyl = - dot(dot(sigma2d(u, p), v), n) * x[0] * ds(2)
b3_cyl = -dot(dot(sigma2d(u, p), v), n) * x[0] * ds(3)
b4_cyl = + dot(p*n, v) * x[0] * ds(4)
b_cyl = b0_cyl + b2_cyl + b3_cyl + b4_cyl

F_cyl = a_cyl + L_cyl + b_cyl

dw = TrialFunction(W)
J = derivative(F, w, dw)
J_cyl = derivative(F_cyl, w, dw)
vtkfile_u = File("Results/velocityTemp" + Texp + ".pvd")

vtkfile_p = File("Results/pressureTemp" + Texp + ".pvd")

vtkfile_mu = File("Results/ViscosityTemp"+Texp+".pvd")
#Y = (Gamma*(1-(x[0]/asp)))
#u0 = (((uin - q) * (exp(2-(1-x[1])) - exp((1-x[1]))))/(exp(2)-1) + q)
#du0dx = ((uin-q)*(-exp(2-(1-x[1]))-exp((1-x[1])))/(exp(2)-1))

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
    (u, p) = w.split()
    vtkfile_u << (u, Gamma_val)
    vtkfile_p << (p, Gamma_val)
    W2 = FunctionSpace(mesh, Q)
    Pmu = project(mu(), W2)
    vtkfile_mu << (Pmu, Gamma_val)

    u_asymp = Expression(('-(x[0]/asp)*((uin-q)*(-exp(2-(1-x[1]))-exp((1-x[1])))/(exp(2)-1)) +'
                          ' (1+2*(Gamma*(1-(x[0]/asp))))*exp(-2*(Gamma*(1-(x[0]/asp))))*((uin-q)*(-exp(2-(1-x[1]))-exp((1-x[1])))/(exp(2)-1))',
                          'Gamma*4*exp(-2*(Gamma*(1-(x[0]/asp))))*(Gamma*(1-(x[0]/asp)))*(q-(((uin - q) * (exp(2-(1-x[1])) - exp((1-x[1]))))/(exp(2)-1) + q)) + '
                          ' (((uin - q) * (exp(2-(1-x[1])) - exp((1-x[1]))))/(exp(2)-1) + q) + exp(-2*(Gamma*(1-(x[0]/asp))))*'
                          '((q-(((uin - q) * (exp(2-(1-x[1])) - exp((1-x[1]))))/(exp(2)-1) + q))*(-10*(Gamma*(1-(x[0]/asp)))+'
                          '4*(Gamma*(1-(x[0]/asp)))+4*(Gamma*(1-(x[0]/asp)))*(Gamma*(1-(x[0]/asp)))*(Gamma*(1-(x[0]/asp)))) +'
                          ' (1-(((uin - q) * (exp(2-(1-x[1])) - exp((1-x[1]))))/(exp(2)-1) + q))*(1-2*(Gamma*(1-(x[0]/asp)))))'),
                         Gamma=Gamma_val, q=1.0, uin=2.0, asp = 0.2, degree=5)

# Let's see if we can actually plot the asymptotic prediction

