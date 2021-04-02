from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

mesh = RectangleMesh(Point(0, 0), Point(0.2, 1), 3, 3)
n = FacetNormal(mesh)
# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2)
Q1 = FiniteElement("CG", triangle, 1)
Q2 = FiniteElement("CG", triangle, 1)
W = FunctionSpace(mesh, MixedElement([V, Q1, Q2]))

# Define Function and TestFunction(s)
w = Function(W)
(u, p, T) = split(w)
(v, q1, q2) = split(TestFunction(W))

Gamma = Constant(5.0)
Pe = Constant(27.0)
# Bi = Constant(58.0)
Bi = Constant(11.6)
# Qc2 = Expression("(1/(x1-x2))*(x[1]<x1)*(x[1]>x2)", degree=2,  x1=0.3, x2=0.1)
# Qfun=2.3710
Qc2 = Expression("Qfun*exp ( -pow( x[1] -(( x1-x2 )/2 + x2), 2 )/( 2*pow( x1-x2,2 ) ) )", degree=1, Qfun=2.3710, x1=0.3,
                 x2=0.1)

Ta = Expression("1-x[1]", degree=1)

# Define the viscosity and bcs
u_in = Constant(-2.0)
u_c = Constant(-1.0)

# Note, x[0] is r and x[1] is x, and x[1] == 0 is the bottom.
inflow = 'near(x[1], 1.0) && x[0]<=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'
bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcT_inflow = DirichletBC(W.sub(2), 0.0, inflow)
bcs = [bcu_inflow, bcu_wall, bcT_inflow, bcu_symmetry, bcu_outflow]
# Define the variational form
epsilon = sym(grad(u))
f = Constant((0, -1))

colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
colors.set_all(0)  # default to zero
colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
colors.set_all(0)  # default to zero

# We match the colours to the defined sketch in the Fenics chapter
CompiledSubDomain("near(x[0], 0.0)").mark(colors, 4)
CompiledSubDomain("near(x[1], 1.0) && x[0]<=0.1").mark(colors, 0)
CompiledSubDomain("near(x[1], 1.0) && x[0]>=0.1").mark(colors, 1)
CompiledSubDomain("near(x[0], 0.2)").mark(colors, 2)  # wall
CompiledSubDomain("near(x[1], 0.0)").mark(colors, 3)  # outflow


def mu(T_local=None):
    if T_local==None:
        T_local = T
    return exp(-Gamma*T_local)


# Create the measure
ds = Measure("ds", subdomain_data=colors)

I = Identity(2)
F = (2 * mu() * inner(epsilon, grad(v)) - div(u) * q1 - div(v) * p - dot(f, v) + dot(u, grad(T)) * q2 + (
       1 / Pe) * inner(grad(q2), grad(T)) - Qc2*q2) * dx \
  - (1 / Pe) * (-Bi * q2 * T * ds(2) + q2 * Bi * Ta * ds(2)) \
  - dot(dot(epsilon, v), n) * ds(0) + dot(dot(p * I, v), n) * ds(0) - dot(dot(epsilon, v), n) * ds(2) \
    + dot(dot(p * I, v), n) * ds(2) - dot(dot(epsilon, v), n) * ds(3) + dot(dot(p * I, v), n) * ds(3)

for Gamma_val in [1, 5, 10, 15, 20, 23]:
    Gamma.assign(Gamma_val)
    print('Gamma =', Gamma_val)
    solve(F == 0, w, bcs)

(u, p, T) = w.split()

N = 5 # number of mesh refinements
h_prev = mesh.hmin()
hvalues = [h_prev]
errors_u = [1]
errors_p = [1]
errors_T = [1]
for i in range(N):
    print(i)
    # Create mesh
    mesh = refine(mesh)
    n = FacetNormal(mesh)
    hvalues.append(mesh.hmin())
    V2 = FunctionSpace(mesh, V)
    Qp = FunctionSpace(mesh, Q1)
    QT = FunctionSpace(mesh, Q2)
    u_prev = interpolate(u, V2) # interpolating values on mesh i + 1
    p_prev = interpolate(p, Qp)
    T_prev = interpolate(T, QT)
    # Define Function and TestFunction(s)
    # Define Taylor--Hood function space W
    V = VectorElement("CG", triangle, 2)
    Q1 = FiniteElement("CG", triangle, 1)
    Q2 = FiniteElement("CG", triangle, 1)
    W = FunctionSpace(mesh, MixedElement([V, Q1, Q2]))

    w = Function(W)
    (u, p, T) = split(w)
    (v, q1, q2) = split(TestFunction(W))

    # Define the viscosity and bcs
    bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
    bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
    bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
    bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
    bcT_inflow = DirichletBC(W.sub(2), 0.0, inflow)
    bcs = [bcu_inflow, bcu_wall, bcT_inflow, bcu_symmetry, bcu_outflow]
    # Define the variational form
    epsilon = sym(grad(u))

    colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    colors.set_all(0)  # default to zero
    colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    colors.set_all(0)  # default to zero

    # We match the colours to the defined sketch in the Fenics chapter
    CompiledSubDomain("near(x[0], 0.0)").mark(colors, 4)
    CompiledSubDomain("near(x[1], 1.0) && x[0]<=0.1").mark(colors, 0)
    CompiledSubDomain("near(x[1], 1.0) && x[0]>=0.1").mark(colors, 1)
    CompiledSubDomain("near(x[0], 0.2)").mark(colors, 2)  # wall
    CompiledSubDomain("near(x[1], 0.0)").mark(colors, 3)  # outflow

    # Create the measure
    ds = Measure("ds", domain=mesh, subdomain_data=colors)

    F = (2 * mu() * inner(epsilon, grad(v)) - div(u) * q1 - div(v) * p - dot(f, v) + dot(u, grad(T)) * q2 + (
           1 / Pe) * inner(grad(q2), grad(T)) - Qc2*q2) * dx \
      - (1 / Pe) * (-Bi * q2 * T * ds(2) + q2 * Bi * Ta * ds(2)) \
      - dot(dot(epsilon, v), n) * ds(0) + dot(dot(p * I, v), n) * ds(0) - dot(dot(epsilon, v), n) * ds(2) \
        + dot(dot(p * I, v), n) * ds(2) - dot(dot(epsilon, v), n) * ds(3) + dot(dot(p * I, v), n) * ds(3)

    for Gamma_val in [1, 5, 10, 15, 20, 23]:
        Gamma.assign(Gamma_val)
        print('Gamma =', Gamma_val)
        solve(F == 0, w, bcs)

    (u, p, T) = w.split()
    u_next = interpolate(u, V2)
    p_next = interpolate(p, Qp)
    T_next = interpolate(T, QT)
    errors_u.append(np.sqrt(assemble(inner(u_next-u_prev, u_next-u_prev)*dx)))
    errors_p.append(np.sqrt(assemble(inner(p_next - p_prev, p_next - p_prev) * dx)))
    errors_T.append(np.sqrt(assemble(inner(T_next - T_prev, T_next - T_prev) * dx)))

rvalues = [0]
for i in range(len(errors_u)-1):
    rvalues.append(np.log(errors_u[i+1]/errors_u[i])/np.log(hvalues[i+1]/hvalues[i]))
print(rvalues)

values = np.asarray([hvalues, errors_u, rvalues])
np.savetxt("Results/ErrorsVelocitycoupled.csv", values.T, delimiter='\t')

rvalues = [0]
for i in range(len(errors_p)-1):
    rvalues.append(np.log(errors_p[i+1]/errors_p[i])/np.log(hvalues[i+1]/hvalues[i]))
print(rvalues)

values = np.asarray([hvalues, errors_p, rvalues])
np.savetxt("Results/ErrorsPressurecoupled.csv", values.T, delimiter='\t')

rvalues = [0]
for i in range(len(errors_T)-1):
    rvalues.append(np.log(errors_T[i+1]/errors_T[i])/np.log(hvalues[i+1]/hvalues[i]))
print(rvalues)

values = np.asarray([hvalues, errors_T, rvalues])
np.savetxt("Results/ErrorsTemperaturecoupled.csv", values.T, delimiter='\t')
