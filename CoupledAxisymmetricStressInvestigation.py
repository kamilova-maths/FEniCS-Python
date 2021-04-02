from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

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


# these should be the same for cylindric and Cartesian... I think ... x.x
# def epsilon2d(v):
#     return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
#                           [v[1].dx(0), v[1].dx(1)]]))


# def sigma2d(v, p):
#     return 2*mu()*epsilon2d(v) - p*Identity(2)


def div_cyl(v):
    return (1/x[0])*(x[0]*v[0]).dx(0) + v[1].dx(1)

Gamma = Constant(10.0)
Pe = Constant(27.0)
Bi = Constant(11.6)
# Cartesian
u_in = Constant(-2.0)

# Cylindric
u_in_cyl = Constant(-4.0)

u_c = Constant(-1.0)


f = Constant((0, -1))

# Choose a mesh. You can also import a previously adapted mesh

#domain = Polygon([Point(0.2, 0), Point(0.2, 1), Point(0.1, 1), Point(0, 1), Point(0, 0)])
#mesh = generate_mesh(domain, 8)
mesh_new = RectangleMesh(Point(0, 0), Point(0.2, 1), 100, 100)
#mesh = Mesh('Meshes/CoupledRefinedMeshGamma5Cartesian.xml')


# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2)
Q = FiniteElement("CG", triangle, 1)
S = FiniteElement("CG", triangle, 1)


inflow = 'near(x[1], 1.0) && x[0]<=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'

iterations = 1
stress_values = []
for i in range(iterations):
    print('Iteration = ', i)
    mesh = mesh_new
    n = FacetNormal(mesh)

    W = FunctionSpace(mesh, MixedElement([V, Q, S]))

    # Define Function and TestFunction(s)
    w = Function(W)
    (u, p, T) = split(w)
    (v, q, s1) = split(TestFunction(W))
    x = SpatialCoordinate(mesh)

    # Define the viscosity and bcs
    # Qfun should be 2.3710
    Qc2 = Expression("Qfun*exp ( -pow( x[1] -(( x1-x2 )/2 + x2), 2 )/( 2*pow( x1-x2,2 ) ) )", degree=3, Qfun=2.3710, x1=0.3,
                    x2=0.1)

    Ta = Expression("1-x[1]", degree=1)

    bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
    bcu_inflow_cyl = DirichletBC(W.sub(0), (0.0, u_in_cyl), inflow)
    bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
    bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
    bcP_outflow = DirichletBC(W.sub(1), 0.0, outflow)
    bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
    bcT_inflow = DirichletBC(W.sub(2), 0.0, inflow)
    bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_symmetry, bcT_inflow]
    bcs_cyl = [bcu_inflow_cyl, bcu_wall, bcu_outflow, bcu_symmetry, bcT_inflow]


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
    b0 = - dot(dot(sigma(u, p), v), n) * ds(0) - s1*dot(grad(T), n)*ds(0)
    b2 = - dot(dot(sigma(u, p), v), n) * ds(2)
    b3 = -dot(dot(sigma(u, p), v), n) * ds(3)
    b4 = + dot(p*n, v) * ds(4)
    L = (- dot(f, v) - Qc2*s1) * dx() - (1/Pe) * (s1 * Bi * Ta * ds(2))

    F = a + L + b0 + b2 + b3 + b4

    # Cylindric
    a_cyl = (inner(sigma_cyl(u, p), epsilon_cyl(v)) - div_cyl(u) * q + (dot(u, grad(T)) * s1 + (
           1 / Pe) * inner(grad(s1), grad(T)))) * x[0] * dx() - (1 / Pe) * (-Bi * s1 * T * x[0] * ds(2))
    b0 = - dot(dot(sigma(u, p), v), n) * x[0] *ds(0) - s1*dot(grad(T), n)*x[0]*ds(0)
    b2 = - dot(dot(sigma(u, p), v), n) * ds(2)
    b3 = -dot(dot(sigma(u, p), v), n) * x[0] * ds(3)
    b4 = + dot(p*n, v) * ds(4)
    L_cyl = (- dot(f, v) - Qc2*s1) * x[0] * dx() - (1/Pe) * (s1 * Bi * Ta * x[0] * ds(2))

    F_cyl = a_cyl + L_cyl + b0 + b2 + b3 + b4

    dw = TrialFunction(W)
    J = derivative(F, w, dw)
    J_cyl = derivative(F_cyl, w, dw)

    for Gamma_val in [1, 10, 15, 20, 23]:
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
    # This computes the INTEGRAL of the normal stress.
    Vsig = TensorFunctionSpace(mesh, "DG", degree=2)
    sig = Function(Vsig, name="Stress")
    sig.assign(project(sigma(u, p), Vsig))
    area1 = assemble(1.0 * ds(1))
    normal_stress1 = assemble(inner(sig * n, n) * ds(1)) / area1
    stress_values.append(normal_stress1)
    mesh_old = mesh
    mesh_new = refine(mesh_old)
print("The normal stress at boundary 1 as mesh is refined is given by", stress_values)
total_normal_stress = assemble(inner(sig * n, n) * ds)

# Compute stress tensor
sigma2 = 2 * mu() * grad(u) - p*Identity(len(u))

# Compute surface traction
T = -sigma2*n

# Compute normal and tangential components
Tn = inner(T,n) # scalar valued
Tt = T - Tn*n # vector valued

# Piecewise constant test functions
scalar = FunctionSpace(mesh, "DG", 2)
vector = VectorFunctionSpace(mesh, "DG", 0)
v1 = TestFunction(scalar)
w1 = TestFunction(vector)

# Assemble piecewise constant functions for stress
normal_stress = Function(scalar)
shear_stress = Function(vector)

Ln = (1 / FacetArea(mesh))*v1*Tn*ds
Lt = (1 / FacetArea(mesh))*inner(w1, Tt)*ds
assemble(Ln, tensor=normal_stress.vector())
assemble(Lt, tensor=shear_stress.vector())
File("Results/NormalStressCartesianCase1.pvd") << normal_stress


# area0 = assemble(1.0*ds(0))
# area1 = assemble(1.0 * ds(1))
# area2 = assemble(1.0 * ds(2))
# area3 = assemble(1.0 * ds(3))
# area4 = assemble(1.0 * ds(4))
#
# Omega0 = assemble((1/area0)*v1*Tn*ds(0))
# Omega1 = assemble((1/area1)*v1*Tn*ds(1))
# Omega2 = assemble((1/area2)*v1*Tn*ds(2))
# Omega3 = assemble((1/area3)*v1*Tn*ds(3))
# Omega4 = assemble((1/area4)*v1*Tn*ds(4))
#
# np.savetxt("Results/CartesianOmega0.csv", np.asarray(Omega0).T, delimiter='\t')
# np.savetxt("Results/CartesianOmega1.csv", np.asarray(Omega1).T, delimiter='\t')
# np.savetxt("Results/CartesianOmega2.csv", np.asarray(Omega2).T, delimiter='\t')
# np.savetxt("Results/CartesianOmega3.csv", np.asarray(Omega3).T, delimiter='\t')
# np.savetxt("Results/CartesianOmega4.csv", np.asarray(Omega4).T, delimiter='\t')