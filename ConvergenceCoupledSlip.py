from fenics import *
from dolfin import *
from dolfin_dg import StokesNitscheBoundary, tangential_proj
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

u_in = Constant(-2.0)
u_c = Constant(-1.0)
u_soln = Constant((0.0, 0.0)) # This is what is dotted with the normal component, and it tells me what the normal
    # component of the velocity is. Namely, if this is zero, then I am setting the NORMAL component of the velocity equal to zero.
Gamma = Constant(23)


# Viscous flux operator
def F_v(u, grad_u, p_local=None):
    if p_local is None:
        p_local = p
    return 2 * eta() * sym(grad_u) - p_local * Identity(2)


# Viscosity model
def eta(T_local=None):
    if T_local is None:
        T_local = T
    return exp(-Gamma * T_local)



Ta = Expression("1-x[1]", degree=1)
Pe = Constant(27.0)
Bi = Constant(11.6)

f = Constant((0.0, -1.0))

g_tau = Constant((0.0, 0.0))  # Tangential stress component


mesh = RectangleMesh(Point(0, 0), Point(0.2, 1), 5, 5, "crossed")
n = FacetNormal(mesh)
# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2)
Q1 = FiniteElement("CG", triangle, 1)
Q2 = FiniteElement("CG", triangle, 1)
W = FunctionSpace(mesh, MixedElement([V, Q1, Q2]))

# Define Function and TestFunction(s)
w = Function(W)
(u, p, T) = split(w)
(v, q, S) = split(TestFunction(W))

inflow = 'near(x[1], 1.0) && x[0]<=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'

bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcT_inflow = DirichletBC(W.sub(2), 0.0, inflow)
bcP_outflow = DirichletBC(W.sub(1), 0.0, outflow)
bcs = [bcu_inflow, bcu_wall, bcP_outflow, bcu_symmetry, bcT_inflow]

colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
colors.set_all(0)  # default to zero


# We match the colours to the defined sketch in the Fenics chapter

CompiledSubDomain("near(x[1], 1.0) && x[0]<=0.1").mark(colors, 0)
CompiledSubDomain("near(x[1], 1.0) && x[0]>=0.1").mark(colors, 1)
CompiledSubDomain("near(x[0], 0.2)").mark(colors, 2)  # wall
CompiledSubDomain("near(x[1], 0.0)").mark(colors, 3)  # outflow
CompiledSubDomain("near(x[0], 0.0)").mark(colors, 4)

ds = Measure("ds", domain=mesh, subdomain_data=colors)

Qc2 = Expression("Qfun*exp(-pow( x[1] -( x1-x2 )/2, 2 )/( 2*pow( x1-x2,2 ) ) )", degree=1, Qfun=0.25, x1=0.5, x2=0.1)

N = inner(F_v(u, grad(u)), grad(v)) * dx - dot(f, v) * dx - div(u) * q * dx + dot(u, grad(T))*S*dx +\
    (1/Pe) * inner(grad(S), grad(T)) * dx - Qc2*S*dx
u_bt = - dot(dot(F_v(u, grad(u), p), v), n)*ds(3) - dot(dot(F_v(u, grad(u), p), v), n)*ds(2) \
    - dot(dot(F_v(u, grad(u), p), v), n)*ds(0)
T_bt = - (1/Pe) * ( - Bi * S * T * ds(2) + S * Bi * Ta * ds(2))
N += T_bt
N += u_bt
# Slip boundary conditions
stokes_nitsche = StokesNitscheBoundary(F_v, u, p, v, q)
N += stokes_nitsche.slip_nitsche_bc_residual(u_soln, g_tau, ds(1))
solve(N == 0, w, bcs, solver_parameters={"newton_solver":{"relative_tolerance":1e-9}})
Qc2 = Expression("Qfun*exp ( -pow( x[1] -( x1-x2 )/2, 2 )/( 2*pow( x1-x2,2 ) ) )", degree=1, Qfun=2.3710, x1=0.3,
              x2=0.1)

N = inner(F_v(u, grad(u)), grad(v)) * dx - dot(f, v) * dx  - div(u) * q * dx + dot(u, grad(T))*S*dx +\
    (1/Pe) * inner(grad(S), grad(T)) * dx - Qc2*S*dx
u_bt = - dot(dot(F_v(u, grad(u), p), v), n)*ds(3) - dot(dot(F_v(u, grad(u), p), v), n)*ds(2) \
    - dot(dot(F_v(u, grad(u), p), v), n)*ds(0)
T_bt = - (1/Pe) * ( - Bi * S * T * ds(2) + S * Bi * Ta * ds(2))
N += T_bt
N += u_bt
# Slip boundary conditions
stokes_nitsche = StokesNitscheBoundary(F_v, u, p, v, q)
N += stokes_nitsche.slip_nitsche_bc_residual(u_soln, g_tau, ds(1))

solve(N == 0, w, bcs, solver_parameters={"newton_solver":{"relative_tolerance":1e-9}})


(u, p, T) = w.split()

N = 4 # number of mesh refinements
h_prev = mesh.hmin()
hvalues = [h_prev]
errors_u = [1]
errors_p = [1]
errors_T = [1]
Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
sig_num = Function(Vsig, name="Stress Numeric")
sig_num.assign(project(F_v(u, grad(u), p), Vsig))
area1 = assemble(1.0 * ds(1))
normal_stress_average = [assemble(inner(sig_num * n, n) * ds(1)) / area1]

vtkfile_u = File('Results/CoupledSlip_meshref_u.pvd')

vtkfile_p = File('Results/CoupledSlip_meshref_p.pvd')

vtkfile_T = File('Results/CoupledSlip_meshref_T.pvd')

vtkfile_stress = File('Results/CoupledSlip_meshref_stress.pvd')

vtkfile_stress << (sig_num, 0)
vtkfile_u << (u, 0)
vtkfile_p << (p, 0)
vtkfile_T << (T, 0)


for i in range(N):
    print(i)
    mesh= refine(mesh)

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
    # Variational formulation
    u, p, T = split(w)
    v, q, S = split(TestFunction(W))

    bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
    bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
    bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
    bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
    bcT_inflow = DirichletBC(W.sub(2), 0.0, inflow)
    bcP_outflow = DirichletBC(W.sub(1), 0.0, outflow)
    bcs = [bcu_inflow, bcu_wall, bcP_outflow, bcu_symmetry, bcT_inflow]

    colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    colors.set_all(0)  # default to zero
    # We match the colours to the defined sketch in the Fenics chapter

    CompiledSubDomain("near(x[1], 1.0) && x[0]<=0.1").mark(colors, 0)
    CompiledSubDomain("near(x[1], 1.0) && x[0]>=0.1").mark(colors, 1)
    CompiledSubDomain("near(x[0], 0.2)").mark(colors, 2)  # wall
    CompiledSubDomain("near(x[1], 0.0)").mark(colors, 3)  # outflow
    CompiledSubDomain("near(x[0], 0.0)").mark(colors, 4)

    # Create the measure
    ds = Measure("ds", domain=mesh, subdomain_data=colors)
    Qc2 = Expression("Qfun*exp(-pow( x[1] -( x1-x2 )/2, 2 )/( 2*pow( x1-x2,2 ) ) )", degree=1, Qfun=0.25, x1=0.5, x2=0.1)
    N = inner(F_v(u, grad(u)), grad(v)) * dx - dot(f, v) * dx - div(u) * q * dx + dot(u, grad(T))*S*dx +\
        (1/Pe) * inner(grad(S), grad(T)) * dx - Qc2*S*dx
    u_bt = - dot(dot(F_v(u, grad(u), p), v), n)*ds(3) - dot(dot(F_v(u, grad(u), p), v), n)*ds(2) \
        - dot(dot(F_v(u, grad(u), p), v), n)*ds(0)
    T_bt = - (1/Pe) * ( - Bi * S * T * ds(2) + S * Bi * Ta * ds(2))
    N += T_bt
    N += u_bt
    # Slip boundary conditions
    stokes_nitsche = StokesNitscheBoundary(F_v, u, p, v, q)
    N += stokes_nitsche.slip_nitsche_bc_residual(u_soln, g_tau, ds(1))
    solve(N == 0, w, bcs, solver_parameters={"newton_solver":{"relative_tolerance":1e-9}})
    Qc2 = Expression("Qfun*exp ( -pow( x[1] -( x1-x2 )/2, 2 )/( 2*pow( x1-x2,2 ) ) )", degree=1, Qfun=2.3710, x1=0.3,
                  x2=0.1)

    N = inner(F_v(u, grad(u)), grad(v)) * dx - dot(f, v) * dx - div(u) * q * dx + dot(u, grad(T))*S*dx +\
        (1/Pe) * inner(grad(S), grad(T)) * dx - Qc2*S*dx
    u_bt = - dot(dot(F_v(u, grad(u), p), v), n)*ds(3) - dot(dot(F_v(u, grad(u), p), v), n)*ds(2) \
        - dot(dot(F_v(u, grad(u), p), v), n)*ds(0)
    T_bt = - (1/Pe) * ( - Bi * S * T * ds(2) + S * Bi * Ta * ds(2))
    N += T_bt
    N += u_bt
    # Slip boundary conditions
    stokes_nitsche = StokesNitscheBoundary(F_v, u, p, v, q)
    N += stokes_nitsche.slip_nitsche_bc_residual(u_soln, g_tau, ds(1))

    solve(N == 0, w, bcs, solver_parameters={"newton_solver":{"relative_tolerance":1e-9}})

    # Plot solutions
    (u, p, T) = w.split()
    vtkfile_u << (u, i + 1)
    vtkfile_p << (p, i + 1)
    vtkfile_T << (T, i + 1)
    u_next = interpolate(u, V2)
    p_next = interpolate(p, Qp)
    T_next = interpolate(T, QT)
    errors_u.append(np.sqrt(assemble(inner(u_next - u_prev, u_next - u_prev) * dx)))
    errors_p.append(np.sqrt(assemble(inner(p_next - p_prev, p_next - p_prev) * dx)))
    errors_T.append(np.sqrt(assemble(inner(T_next - T_prev, T_next - T_prev) * dx)))
    Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
    sig_num = Function(Vsig, name="Stress Numeric")
    sig_num.assign(project(F_v(u, grad(u), p), Vsig))
    area1 = assemble(1.0 * ds(1))
    normal_stress_average.append(assemble(inner(sig_num * n, n) * ds(1)) / area1)

values = np.asarray([hvalues, errors_u, errors_p, errors_T, normal_stress_average])
np.savetxt("Results/ErrorsConvergenceCoupledSlip.csv", values.T, delimiter='\t')
