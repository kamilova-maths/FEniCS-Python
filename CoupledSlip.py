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
lo = 0.9
hi = 1
tol = 0.0001
count = 0
a = 1

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


# Viscosity model
def eta(T_local=None):
    if T_local is None:
        T_local = T
    return exp(-Gamma * T_local)


total_stress_old = 0.0
stress_array = []
Na = 1
da = (hi-lo)/Na
a_values = []
Ta = Expression("1-x[1]", degree=1)
Pe = Constant(27.0)
Bi = Constant(11.6)
Qc = Constant(0.0)

f = Constant((0.0, -1.0))

g_tau = Constant((0.0, 0.0))  # Tangential stress component
for i in range(0, Na):
    print('a is ', a)
    a_values.append(a)

    # Geometry
    abnd = str(a)
    #domain = Polygon([Point(0.2, 0), Point(0.2, a), Point(0.1, 1), Point(0, 1), Point(0, 0)])
    mesh = RectangleMesh(Point(0, 0), Point(0.2, 1), 100, 100)

    n = FacetNormal(mesh)
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
    bcs = [bcu_inflow, bcu_wall, bcP_outflow, bcu_symmetry, bcT_inflow]

    colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    colors.set_all(0)  # default to zero
    # We match the colours to the defined sketch in the Fenics chapter
    CompiledSubDomain("near(x[0], 0.0)").mark(colors, 4)
    CompiledSubDomain("near(x[1], 1.0) && x[0]<=0.1").mark(colors, 0)
    CompiledSubDomain("near( ( ("+abnd+"-1) /0.1)*(x[0] - 0.2) +" + abnd + "- x[1], 0.0) && x[0]>=0.1").mark(colors, 1)
    CompiledSubDomain("near(x[0], 0.2)").mark(colors, 2)  # wall
    CompiledSubDomain("near(x[1], 0.0)").mark(colors, 3)  # outflow
    # Create the measure
    ds = Measure("ds", domain = mesh, subdomain_data=colors)

    # Variational formulation
    u, p, T = split(w)
    v, q, S = split(TestFunction(W))

    Qc2 = Expression("Qfun*exp(-pow( x[1] -( x1-x2 )/2, 2 )/( 2*pow( x1-x2,2 ) ) )", degree=1, Qfun=0.25, x1=0.5, x2=0.1)

    N = inner(F_v(u, grad(u)), grad(v)) * dx - dot(f, v) * dx + div(u) * q * dx + dot(u, grad(T))*S*dx +\
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

    N = inner(F_v(u, grad(u)), grad(v)) * dx - dot(f, v) * dx  + div(u) * q * dx + dot(u, grad(T))*S*dx +\
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

    # Compute stress tensor
    sigma_expr = 2 * eta() * grad(u) - p * Identity(len(u))

    # Compute surface traction
    Tr = -sigma_expr * n

    # Compute normal and tangential components
    Tn = inner(Tr, n)
    # scalar valued
    Tt = Tr - Tn * n

    # Piecewise constant test functions
    scalar = FunctionSpace(mesh, "CG", 2) # the original one was "DG"
    v1 = TestFunction(scalar)
    # w1 = TestFunction(vector)

    # Assemble piecewise constant functions for stress
    normal_stress = Function(scalar)
    Ln = (1 / FacetArea(mesh)) * v1 * Tn * ds # could also be the specific boundary with ds(1) (for example)

    # Ln = v1*Tn*ds(subdomain_data=colors, subdomain_id=1)/(Constant(1.0)*ds(subdomain_data=colors, subdomain_id=1))
    # Lt = (1 / FacetArea(mesh))*inner(w1,Tt)*ds(1)
    assemble(Ln, tensor=normal_stress.vector())
    File("Results/NormalStressCartesianCase2.pvd") << normal_stress
    # Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
    # sig = Function(Vsig, name="Stress")
    # sig.assign(project(F_v(u, grad(u)), Vsig))
    # area0 = assemble(1.0*ds(0))
    # area1 = assemble(1.0 * ds(1))
    # area2 = assemble(1.0 * ds(2))
    # area3 = assemble(1.0 * ds(3))
    # area4 = assemble(1.0 * ds(4))
    # print("area at ds0 is", area0)
    # print("area at ds1 is", area1)
    # print("area at ds2 is", area2)
    # print("area at ds3 is", area3)
    # print("area at ds4 is", area4)
    # normal_stress0 = assemble(inner(sig*n, n)*ds(0))/area0
    # normal_stress1 = assemble(inner(sig * n, n) * ds(1))/area1
    # normal_stress2 = assemble(inner(sig * n, n) * ds(2))/area2
    # normal_stress3 = assemble(inner(sig * n, n) * ds(3))/area3
    # normal_stress4 = assemble(inner(sig * n, n) * ds(4))/area4
    # print("Stress at (0.1, 1):", sig(0.1, 1))
    # print("Normal stress at boundary 0", normal_stress0)
    # print("Normal stress at boundary 1", normal_stress1)
    # print("Normal stress at boundary 2", normal_stress2)
    # print("Normal stress at boundary 3", normal_stress3)
    # print("Normal stress at boundary 4", normal_stress4)
    # #pds0 = assemble(dot(p*n, n), ds(0))
    # #print("normal pressure at ds0 is", pds0)

    a = a - da


see = 0.0
if see == 1.0:

    c = plot(u, title='Velocity')
    plt.colorbar(c)
    plt.show()
    # #

    c = plot(T, title='Temperature')
    plt.colorbar(c)
    plt.show()


    c = plot(normal_stress, title='Normal stress on domain')
    plt.colorbar(c)
    plt.show()

    c = plot(p, title='pressure')
    plt.colorbar(c)
    plt.show()

sav = 1.0
if sav == 1.0:
    File("Results/velocityCoupledFreeSlip.pvd") << u
    File("Results/TemperatureCoupledFreeSlip.pvd") << T
    File("Results/NormalStressCoupledFreeSlip.pvd") << normal_stress
    File("Results/pressureCoupledFreeSlip.pvd") << p


