from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import ufl
# Define mesh and geometry - We solve for half of the domain we need, and impose symmetry
#domain = Polygon([Point(1, 0), Point(1, 0.5), Point(0.5, 1), Point(0, 1), Point(0, 0)])
#mesh = generate_mesh(domain, 100)


def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1), 0],
                          [v[1].dx(0), v[1].dx(1), 0],
                          [0, 0, 0]]))


# stress tensor
def sigma(v, p):
    return 2*mu*epsilon(v)-Id(p)


def Id(p):
    return as_tensor([[p, 0, 0],
                      [0, p, 0],
                     [0, 0, p]])


def cond(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


def sigmabc(v, p):
    return 2*mu*cond(v) - p*Identity(2)


mesh = RectangleMesh(Point(0, 0), Point(0.2, 1.0), 3, 3)
#mesh = Mesh('Meshes/IsothermalRefinedMesh.xml')

n = FacetNormal(mesh)

# V = VectorElement("CG", mesh.ufl_cell(),  2)
# Q = FiniteElement("DG", mesh.ufl_cell(), 0)
# P2 = FiniteElement("CG", mesh.ufl_cell(), 2) # these friends do not work for my chosen boundary conditions.
# B = FiniteElement("Bubble", mesh.ufl_cell(), mesh.topology().dim() + 1)
# V = VectorElement(NodalEnrichedElement(P2, B))
# Q = FiniteElement("DG", mesh.ufl_cell(), 1)

# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2) # original spaces
Q = FiniteElement("CG", triangle, 1) # original spaces
W = FunctionSpace(mesh, MixedElement([V, Q]))

# Define Function and TestFunction(s)
w = Function(W)
(u, p) = split(w)
(v, q) = split(TestFunction(W))


# Manufactured solution of the easy case
solns_ex = ['-sin(2.0*pi*x[1])', 'sin(2.0*pi*x[0])', '-cos(2.0*pi*x[0])*cos(2.0*pi*x[1])']
solns_f = ['2*pi*(cos(2*pi*x[1])*sin(2*pi*x[0]) - 2*pi*sin(2*pi*x[1]))', '2*pi*(2*pi*sin(2*pi*x[0]) + cos(2*pi*x[0])*sin(2*pi*x[1]))']
u_ex = Expression((solns_ex[0], solns_ex[1]), element=V,domain=mesh)
p_ex = Expression(solns_ex[2], element=Q, domain=mesh)
f_ex = Expression((solns_f[0], solns_f[1]), element=V, domain=mesh)
# UFL form for the source term

# u_fun = Expression(('0.0', '-uout*exp(x[1]*(x[0]-0.2))'), degree=3, uout = 1 )
# u_fun = Expression(('0.0', '-(x[0]-0.2)*(1-x[1])*(1-x[1])'), degree=2)
u_fun = Expression(('-sin(2*pi*x[1])', 'sin(2*pi*x[0])'), degree=2)
W2 = FunctionSpace(mesh, V)
u_e = project(u_fun, W2)


#f = Expression(('-uout*exp(x[0]-0.2)*(x[0]+0.8)', ' uout*(x[1]*x[1]+(x[0]-0.2)*(x[0]-0.2))*exp(x[1]*(x[0]-0.2))'), degree=2, uout = 1 )
#f = Expression(('0.0', '2*(x[0]-0.2)'), degree=2)
f = Expression(('2*pi*(cos(2*pi*x[1])*sin(2*pi*x[0]) - 2*pi*sin(2*pi*x[1]))', '2*pi*(2*pi*sin(2*pi*x[0]) + cos(2*pi*x[0])*sin(2*pi*x[1]))'), degree=2)
f_e = project(f, W2)

#g = Expression(' -uout*(x[0]-0.2)*exp(x[1]*(x[0]-0.2))', degree=2, uout = 1 )
g = Expression('2*(1-x[1])*(x[0]-0.2)', degree=2)
W3 = FunctionSpace(mesh, Q)
g_e = project(g, W3) # p is g_e3

# p_e = project(Expression(' -uout*(x[0]-0.2)*exp(x[0]-0.2)', degree=2, uout = 1 ), W3)
p_e = project(Expression('-cos(2*pi*x[0])*cos(2*pi*x[1])', degree=2), W3)

p_right = Expression('cos(2*pi*x[0])', degree=2)
# Define the viscosity and bcs

mu = Constant(1.0)
# mu = Expression('exp(-a*pow(x[0],2))', degree=2, a=10)

#u_in = Constant(-2.0)
#u_in = Expression(('0.0', '-exp(x[0]-0.2)'), degree=2)
#u_in = Expression(('0.0', '-(x[0]-0.2)'), degree=2)
#u_c = Expression(('0.0', '0.0'), degree=2)
#u_c = Constant(-1.0)
u_outflow = Expression(('0.0', 'sin(2*pi*x[0])'), degree=2)
u_inflow = Expression(('0.0', 'sin(2*pi*x[0])'), degree=2 )
u_wall = Expression(('-sin(2*pi*x[1])', 'sin(0.4*pi)'), degree=2 )
u_centre = Expression(('-sin(2*pi*x[1])', '0.0'), degree=2 )
inflow = 'near(x[1], 1.0) && x[0]<=0.1' # This is C++ code.
right = 'near(x[1], 1.0) && x[0]>=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'
bcu_inflow = DirichletBC(W.sub(0), u_ex, inflow)
bcu_wall = DirichletBC(W.sub(0), u_ex, wall)
bcu_outflow = DirichletBC(W.sub(0), u_ex, outflow)
#bcP_outflow = DirichletBC(W.sub(1), Constant(0.0), outflow)
#bcP_inflow = DirichletBC(W.sub(1), Constant(0.0), inflow)
#bcP_right = DirichletBC(W.sub(1), Constant(0.0), right)
bcu_symmetry = DirichletBC(W.sub(0), u_ex, centre)
bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_symmetry]
# Define stress tensor
# epsilon = sym(grad(u))
x = SpatialCoordinate(mesh)
#f1 = as_vector([-exp(x[0]-0.2)*(x[0]+0.8), (x[1]*x[1]+(x[0]-0.2)*(x[0]-0.2))*exp(x[1]*(x[0]-0.2))])
facet_f = MeshFunction("size_t", mesh, mesh.topology().dim() - 1) # FACET function
CompiledSubDomain('near(x[1], 1.0) && x[0]<=0.1').mark(facet_f, 0)
CompiledSubDomain('near(x[1], 1.0) && x[0]>=0.1').mark(facet_f, 1)
CompiledSubDomain('near(x[0], 0.2)').mark(facet_f, 2)  # wall
CompiledSubDomain('near(x[1], 0.0)').mark(facet_f, 3)  # outflow
CompiledSubDomain('near(x[0], 0.0)').mark(facet_f, 4)
# Create the measure
ds = Measure("ds", domain=mesh, subdomain_data=facet_f)
a1 = (inner(sigma(u, p), epsilon(v))) * dx
a2 = (- div(u) * q - dot(f_ex, v)) * dx
#b1 = dot(p*n, v)*ds(0) - dot(mu*dot(nabla_grad(u), n), v)*ds(0) + dot(p*n, v)*ds(2) - dot(mu*dot(nabla_grad(u), n), v)*ds(2) + \
#      dot(p*n, v)*ds(3) - dot(mu*dot(nabla_grad(u), n), v)*ds(3) + dot(p*n, v)*ds(4) - dot(mu*dot(nabla_grad(u), n), v)*ds(4)
b1 = - dot(dot(sigmabc(u, p), n), v)*ds(0) - dot(dot(sigmabc(u, p), n), v)*ds(2) - dot(dot(sigmabc(u, p), n), v)*ds(3) \
    - dot(dot(sigmabc(u, p), n), v)*ds(4) - dot(dot(p_ex*Identity(2), n), v)*ds(1)
F = a1 + a2 + b1
# Solve problem
solve(F == 0, w,  bcs)
(u, p) = w.split()

h_prev = mesh.hmin()
hvalues = [h_prev]
#errors_u = [sqrt(abs(assemble((u_e - u)**2*dx)))]
#errors_p = [sqrt(abs(assemble((p_e - p)**2*dx)))]
errors_u = [errornorm(u_ex, u)]
errors_p = [errornorm(p_ex, p)]

for i in range(6):
    mesh = refine(mesh)
    n = FacetNormal(mesh)
    hvalues.append(mesh.hmin())

    # Define Taylor--Hood function space W
    V = VectorElement("CG", triangle, 2)  # original spaces
    Q = FiniteElement("CG", triangle, 1)  # original spaces
    W = FunctionSpace(mesh, MixedElement([V, Q]))

    # Define Function and TestFunction(s)
    w = Function(W)
    (u, p) = split(w)
    (v, q) = split(TestFunction(W))

    # Manufactured solution of the easy case
    #u_fun = Expression(('0.0', '-uout*exp(x[1]*(x[0]-0.2))'), degree=2, uout=1)
    u_ex = Expression((solns_ex[0], solns_ex[1]), element=V, domain=mesh)
    p_ex = Expression(solns_ex[2], element=Q, domain=mesh)
    f_ex = Expression((solns_f[0], solns_f[1]), element=V, domain=mesh)

    #f1 = as_vector([-exp(x[0]-0.2)*(x[0]+0.8), (x[1]*x[1]+(x[0]-0.2)*(x[0]-0.2))*exp(x[1]*(x[0]-0.2))])

    #g = Expression(' -uout*(x[0]-0.2)*exp(x[1]*(x[0]-0.2))', degree=2, uout=1)

    # W3 = FunctionSpace(mesh, Q)
    # g_e = project(g, W3)  # p is g_e

    #p_e = project(Expression(' -uout*(x[0]-0.2)*exp(x[0]-0.2)', degree=2, uout=1), W3)
    # Define the viscosity and bcs

    mu = Constant(1.0)
    # mu = Expression('exp(-a*pow(x[0],2))', degree=2, a=10)

    # u_in = Constant(-2.0)
    #u_in = Expression(('0.0', '-uout*exp(x[0]-0.2)'), degree=2, uout=1)
    #u_c = Constant(-1.0)

    inflow = 'near(x[1], 1.0) && x[0]<=0.1'  # This is C++ code.
    right = 'near(x[1], 1.0) && x[0]>=0.1'
    wall = 'near(x[0], 0.2)'
    centre = 'near(x[0], 0.0)'
    outflow = 'near(x[1], 0.0)'
    bcu_inflow = DirichletBC(W.sub(0), u_ex, inflow)
    bcu_wall = DirichletBC(W.sub(0), u_ex, wall)
    bcu_outflow = DirichletBC(W.sub(0), u_ex, outflow)
    #bcP_outflow = DirichletBC(W.sub(1), Constant(0.0), outflow)
    #bcP_inflow = DirichletBC(W.sub(1), Constant(0.0), inflow)
    #bcP_right = DirichletBC(W.sub(1), Constant(0.0), right)
    bcu_symmetry = DirichletBC(W.sub(0), u_ex, centre)
    bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_symmetry]
    # Define stress tensor
    # epsilon = sym(grad(u))
    x = SpatialCoordinate(mesh)

    facet_f = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)  # FACET function
    CompiledSubDomain('near(x[1], 1.0) && x[0]<=0.1').mark(facet_f, 0)
    CompiledSubDomain('near(x[1], 1.0) && x[0]>=0.1').mark(facet_f, 1)
    CompiledSubDomain('near(x[0], 0.2)').mark(facet_f, 2)  # wall
    CompiledSubDomain('near(x[1], 0.0)').mark(facet_f, 3)  # outflow
    CompiledSubDomain('near(x[0], 0.0)').mark(facet_f, 4)
    # Create the measure
    ds = Measure("ds", domain=mesh, subdomain_data=facet_f)
    a1 = (inner(sigma(u, p), epsilon(v))) * dx
    a2 = (- div(u) * q - dot(f_ex, v)) * dx
    # b1 = dot(p*n, v)*ds(0) - dot(mu*dot(nabla_grad(u), n), v)*ds(0) + dot(p*n, v)*ds(2) - dot(mu*dot(nabla_grad(u), n), v)*ds(2) + \
    #      dot(p*n, v)*ds(3) - dot(mu*dot(nabla_grad(u), n), v)*ds(3) + dot(p*n, v)*ds(4) - dot(mu*dot(nabla_grad(u), n), v)*ds(4)
    b1 = - dot(dot(sigmabc(u, p), n), v) * ds(0) - dot(dot(sigmabc(u, p), n), v) * ds(2) - dot(dot(sigmabc(u, p), n),
                                                                                               v) * ds(3) \
         - dot(dot(sigmabc(u, p), n), v) * ds(4) - dot(dot(p_ex*Identity(2), n), v) * ds(1)
    F = a1 + a2 + b1
    # Solve problem
    solve(F == 0, w, bcs)
    (u, p) = w.split()
    #errors_u.append(sqrt(abs(assemble((u_e - u)**2*dx))))
    #errors_p.append(sqrt(abs(assemble((p_e - p)**2*dx))))
    errors_u.append(errornorm(u_ex, u))
    errors_p.append(errornorm(p_ex, p))

rvalues = [0]
for i in range(len(errors_u)-1):
    rvalues.append(np.log(errors_u[i+1]/errors_u[i])/np.log(hvalues[i+1]/hvalues[i]))
print(rvalues)

values = np.asarray([hvalues, errors_u, rvalues])
np.savetxt("Results/ErrorsExactVelocityIsothermal.csv", values.T, delimiter='\t')

rvalues = [0]
for i in range(len(errors_p)-1):
    rvalues.append(np.log(errors_p[i+1]/errors_p[i])/np.log(hvalues[i+1]/hvalues[i]))
print(rvalues)

values = np.asarray([hvalues, errors_p, rvalues])
np.savetxt("Results/ErrorsExactPressureIsothermal.csv", values.T, delimiter='\t')

# # Plot solutions
# (u, p) = w.split()
#
# # Compute stress tensor
# sigma = 2 * mu * grad(u) - p*Identity(len(u))
#
# # Compute surface traction
# T = -sigma*n
#
# # Compute normal and tangential components
# Tn = inner(T,n) # scalar valued
# Tt = T - Tn*n # vector valued
#
# # Piecewise constant test functions
# scalar = FunctionSpace(mesh, "DG", 0)
# vector = VectorFunctionSpace(mesh, "DG", 0)
# v1 = TestFunction(scalar)
# w1 = TestFunction(vector)
#
# # Assemble piecewise constant functions for stress
# normal_stress = Function(scalar)
# shear_stress = Function(vector)
#
# Ln = (1 / FacetArea(mesh))*v1*Tn*ds
# Lt = (1 / FacetArea(mesh))*inner(w1, Tt)*ds
# assemble(Ln, tensor=normal_stress.vector())
# assemble(Lt, tensor=shear_stress.vector())

#File("Results/NormalStressIsothermalNoStress.pvd") << normal_stress
# c = plot(normal_stress, title='stress?')
# plt.colorbar(c)
# plt.show()
# error = (u_e - u)**2*dx
# E = sqrt(abs(assemble(error)))
# print('Velocity error first way is ', E)
#
# print('Velocity error second way is', errornorm(u_e, u))
#
# print('Pressure error first way is ', sqrt(abs(assemble((p_e-p)**2*dx))))
# print('Pressure error second way is ', errornorm(p_e, p))
File("Results/uexact.pvd") << u_e
File("Results/unum.pvd") << u

File("Results/pexact.pvd") << p_e

File("Results/pnum.pvd") << p
#
Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
sig = Function(Vsig, name="Stress")
sig.assign(project(sigmabc(u, p), Vsig))
area0 = assemble(1.0 * ds(0))
area1 = assemble(1.0 * ds(1))
area2 = assemble(1.0 * ds(2))
area3 = assemble(1.0 * ds(3))
area4 = assemble(1.0 * ds(4))
# print("area at ds0 is", area0)
# print("area at ds1 is", area1)
# print("area at ds2 is", area2)
# print("area at ds3 is", area3)
# print("area at ds4 is", area4)
normal_stress0 = assemble(inner(sig*n, n)*ds(0))/area0
normal_stress1 = assemble(inner(sig * n, n) * ds(1))/area1
normal_stress2 = assemble(inner(sig * n, n) * ds(2))/area2
normal_stress3 = assemble(inner(sig * n, n) * ds(3))/area3
normal_stress4 = assemble(inner(sig * n, n) * ds(4))/area4
# print("Stress at (0.1, 1):", sig(0.1, 1))
# print("Normal stress at boundary 0", normal_stress0)
# print("Normal stress at boundary 1", normal_stress1)
# print("Normal stress at boundary 2", normal_stress2)
# print("Normal stress at boundary 3", normal_stress3)
# print("Normal stress at boundary 4", normal_stress4)
#
#
