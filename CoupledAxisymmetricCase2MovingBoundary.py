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


# Viscous flux operator
def F_v_cyl(u, grad_u, p_local=None):
    if p_local is None:
        p_local = p
    return 2 * eta() * sym(grad_u) - p_local * Identity(3)


# Viscosity model
def eta(T_local=None):
    if T_local is None:
        T_local = T
    return exp(-Gamma * T_local)


def div_cyl(v):
    return (1/x[0])*(x[0]*v[0]).dx(0) + v[1].dx(1)


def grad_cyl(v):
    return as_tensor([[v[0].dx(0), 0, v[0].dx(1)],
                          [0, v[0] / x[0], 0],
                          [v[1].dx(0), 0, v[1].dx(1)]])



Pe = Constant(27.0)
Bi = Constant(11.6)
Qc = Constant(0.0)

f = Constant((0.0, -1.0))

g_tau = Constant((0.0, 0.0))  # Tangential stress component

mesh = RectangleMesh(Point(0.0), Point(0.2, 1), 60, 60)
x = SpatialCoordinate(mesh)

Xvalues = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2]
x_1 = 0.3
x_2 = 0.1
x_min = 0
x_max = 1
tol = 1e-10

Q_small = Expression("Qfun*exp ( -pow( x[1] -(( x1-x2 )/2 + x2), 2 )/( 2*pow( x1-x2,2 ) ) )", degree=3, Qfun=0.0,
                 x1=0.3,
                 x2=0.1)
Q_large = Expression("Qfun*exp ( -pow( x[1] -(( x1-x2 )/2 + x2), 2 )/( 2*pow( x1-x2,2 ) ) )", degree=3, Qfun=2.3710,
                 x1=0.3,
                 x2=0.1)

normal_stress_values_ds0 = []
normal_stress_values_ds1 = []
normal_stress_values_ds3 = []
coord = "Cyl"
#coord = "Car"

for i in range(len(Xvalues)):
    print("X value is ", Xvalues[i])
    n = FacetNormal(mesh)
    # Function space
    We = MixedElement([VectorElement("CG", mesh.ufl_cell(), 2), FiniteElement("CG", mesh.ufl_cell(), 1),
                       FiniteElement("CG", mesh.ufl_cell(), 1)])
    W = FunctionSpace(mesh, We)
    w = Function(W)
    bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
    bcu_inflow_cyl = DirichletBC(W.sub(0), (0.0, -4.0), inflow)
    bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
    bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
    bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
    bcT_inflow = DirichletBC(W.sub(2), 0.0, inflow)
    bcP_outflow = DirichletBC(W.sub(1), 0.0, outflow)
    bcs = [bcu_inflow, bcu_wall, bcP_outflow, bcu_symmetry, bcT_inflow]
    bcs_cyl = [bcu_inflow_cyl, bcu_wall, bcP_outflow, bcu_symmetry, bcT_inflow]

    colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    colors.set_all(0)  # default to zero
    # We match the colours to the defined sketch in the Fenics chapter
    CompiledSubDomain("near(x[1], 1.0) && x[0]<=0.1").mark(colors, 0)
    CompiledSubDomain('near(x[1], 1.0) && x[0]>=0.1').mark(colors, 1)
    CompiledSubDomain("near(x[0], 0.2)").mark(colors, 2)  # wall
    CompiledSubDomain("near(x[1], 0.0)").mark(colors, 3)  # outflow
    CompiledSubDomain("near(x[0], 0.0)").mark(colors, 4)
    # Create the measure
    ds = Measure("ds", domain=mesh, subdomain_data=colors)

    # Variational formulation
    u, p, T = split(w)
    v, q, S = split(TestFunction(W))

    Q_small.x1 = x_1 + Xvalues[i]
    Q_small.x2 = x_2 + Xvalues[i]
    x_min = 0 + Xvalues[i]
    x_max = 1 + Xvalues[i]
    Ta = conditional(ge(x[1], x_max), tol, conditional(ge(x[1], x_min), 1 - x[1] + Xvalues[i], 1.0 - tol))
    # Cartesian
    # N = inner(F_v(u, grad(u)), grad(v)) * dx - dot(f, v) * dx + div(u) * q * dx + dot(u, grad(T))*S*dx +\
    #     (1/Pe) * inner(grad(S), grad(T)) * dx - Q_small*S*dx
    # u_bt = - dot(dot(F_v(u, grad(u), p), v), n)*ds(3) - dot(dot(F_v(u, grad(u), p), v), n)*ds(2) \
    #     - dot(dot(F_v(u, grad(u), p), v), n)*ds(0) + dot(p*n, v) * ds(4)
    # T_bt = - (1/Pe) * ( - Bi * S * T * ds(2) + S * Bi * Ta * ds(2))
    # N += T_bt
    # N += u_bt
    #
    #Cylindric
    N_cyl = inner(F_v_cyl(u, grad_cyl(u)), grad_cyl(v)) * x[0] * dx \
            - dot(f, v) * x[0] * dx + div_cyl(u) * q * x[0] * dx + dot(u, grad(T)) * S * x[0] * dx \
        + (1/Pe) * inner(grad(S), grad(T)) * x[0] * dx - Q_small * S * x[0] * dx
    #u_bt_cyl = - dot(dot(F_v(u, grad(u), p), v), n)*x[0] * ds(3) - dot(dot(F_v(u, grad(u), p), v), n)*x[0]*ds(2) \
     #   - dot(dot(F_v(u, grad(u), p), v), n)*x[0]*ds(0) + dot(p*n, v) * x[0] * ds(4)

    u_bt_cyl = - dot(dot(F_v(u, grad(u), p), v), n)*x[0]*ds(1) + dot(p*n, v) * x[0] * ds(4)
    T_bt_cyl = - (1/Pe) * (- Bi * S * T * x[0]*ds(2) + S * Bi * Ta * x[0] * ds(2))
    N_cyl += T_bt_cyl
    N_cyl += u_bt_cyl

    # Slip boundary conditions
    # stokes_nitsche = StokesNitscheBoundary(F_v, u, p, v, q)
    # N += stokes_nitsche.slip_nitsche_bc_residual(u_soln, g_tau, ds(1))
    #
    stokes_nitsche_cyl = StokesNitscheBoundary(F_v, u, p, v, q)
    N_cyl += stokes_nitsche_cyl.slip_nitsche_bc_residual(u_soln, g_tau, ds(1))

    #solve(N == 0, w, bcs, solver_parameters={"newton_solver": {"relative_tolerance": 1e-9}})

    solve(N_cyl == 0, w, bcs_cyl, solver_parameters={"newton_solver": {"relative_tolerance": 1e-9}})

    Q_large.x1 = x_1 + Xvalues[i]
    Q_large.x2 = x_2 + Xvalues[i]
    # Cartesian
    # N = inner(F_v(u, grad(u)), grad(v)) * dx - dot(f, v) * dx  + div(u) * q * dx + dot(u, grad(T))*S*dx +\
    #     (1/Pe) * inner(grad(S), grad(T)) * dx - Q_large*S*dx
    # u_bt = - dot(dot(F_v(u, grad(u), p), v), n)*ds(3) - dot(dot(F_v(u, grad(u), p), v), n)*ds(2) \
    #     - dot(dot(F_v(u, grad(u), p), v), n)*ds(0) + dot(p*n, v) * ds(4)
    # T_bt = - (1/Pe) * ( - Bi * S * T * ds(2) + S * Bi * Ta * ds(2))
    # N += T_bt
    # N += u_bt

    #Cylindric
    N_cyl = inner(F_v_cyl(u, grad_cyl(u)), grad_cyl(v)) * x[0] * dx \
            - dot(f, v) * x[0] * dx + div_cyl(u) * q * x[0] * dx + dot(u, grad(T)) * S * x[0] * dx \
            + (1 / Pe) * inner(grad(S), grad(T)) * x[0] * dx - Q_large * S * x[0] * dx
    # u_bt_cyl = - dot(dot(F_v(u, grad(u), p), v), n) * x[0] * ds(3) - dot(dot(F_v(u, grad(u), p), v), n) * x[0] * ds(2) \
    #           - dot(dot(F_v(u, grad(u), p), v), n) * x[0] * ds(0) + dot(p*n, v) * x[0] * ds(4)
    u_bt_cyl = - dot(dot(F_v(u, grad(u), p), v), n) * x[0] * ds(1) + dot(p * n, v) * x[0] * ds(4)

    T_bt_cyl = - (1 / Pe) * (- Bi * S * T * x[0] * ds(2) + S * Bi * Ta * x[0] * ds(2))
    N_cyl += T_bt_cyl
    N_cyl += u_bt_cyl

    # Slip boundary conditions
    # stokes_nitsche = StokesNitscheBoundary(F_v, u, p, v, q)
    # N += stokes_nitsche.slip_nitsche_bc_residual(u_soln, g_tau, ds(1))

    stokes_nitsche_cyl = StokesNitscheBoundary(F_v, u, p, v, q)
    N_cyl += stokes_nitsche_cyl.slip_nitsche_bc_residual(u_soln, g_tau, ds(1))

    # #Cartesian
    #solve(N == 0, w, bcs, solver_parameters={"newton_solver": {"relative_tolerance": 1e-9}})

    #Cylindric
    solve(N_cyl == 0, w, bcs_cyl, solver_parameters={"newton_solver": {"relative_tolerance": 1e-9}})


    # Plot solutions
    (u, p, T) = w.split()
    File("Results/v_CoupledCase2" + coord + "_X" + str(Xvalues[i]) + ".pvd") << u
    File("Results/T_CoupledCase2" + coord + "_X" + str(Xvalues[i]) + ".pvd") << T
    File("Results/p_CoupledCase2" + coord + "_X" + str(Xvalues[i]) + ".pvd") << p
    # Compute stress tensor - cartesian
    Vsig = TensorFunctionSpace(mesh, "DG", degree=1)
    sig = Function(Vsig, name="Stress" + str(Xvalues[i]))
    # Cartesian
    sig.assign(project(F_v(u, grad(u), p), Vsig))
    File("Results/StressCase2" + coord + "_X" + str(Xvalues[i]) + ".pvd") << sig
    # Cartesian
    # area1 = assemble(1.0 * ds(1))
    # normal_stress1 = assemble(inner(sig * n, n) * ds(1)) / area1
    # area0 = assemble(1.0 * ds(0))
    # area3 = assemble(1.0 * ds(3))
    # normal_stress0 = assemble(inner(sig * n, n) * ds(0)) / area0
    # normal_stress3 = assemble(inner(sig * n, n) * ds(3)) / area3

    # Cylindric
    area1 = assemble(1.0 * x[0] * ds(1))
    normal_stress1 = assemble(inner(sig * n, n) * x[0] * ds(1)) / area1
    area0 = assemble(1.0 * x[0] * ds(0))
    area3 = assemble(1.0 * x[0] * ds(3))
    normal_stress0 = assemble(inner(sig * n, n) * x[0] * ds(0)) / area0
    normal_stress3 = assemble(inner(sig * n, n) * x[0] * ds(3)) / area3

    normal_stress_values_ds0.append(normal_stress0)
    normal_stress_values_ds1.append(normal_stress1)
    normal_stress_values_ds3.append(normal_stress3)


values = np.asarray([Xvalues, normal_stress_values_ds0, normal_stress_values_ds1, normal_stress_values_ds3])
np.savetxt("Results/XvsAverageNormalStressTop" + coord + ".csv", values.T, delimiter='\t')
