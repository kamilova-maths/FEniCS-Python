from fenics import *
from dolfin import *
from mshr import *
from dolfin_dg import StokesNitscheBoundary, tangential_proj
import matplotlib.pyplot as plt
import numpy as np

u_in = Constant(-2.0)
u_c = Constant(-1.0)
u_soln = Constant((0.0, 0.0))  # This is what is dotted with the normal component, and it tells me what the normal
# component of the velocity is. Namely, if this is zero, then I am setting the NORMAL component of the velocity equal to zero.

lo = 0.8
hi = 1
tol = 0.0001
count = 0
a = 1

eta = Constant(1.0)
# Note, x[0] is r and x[1] is x, and x[1] == 0 is the bottom.
inflow = 'near(x[1], 1.0) && x[0]<=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'
total_stress_old = 0.0
stress_array = []
Na = 20
da = (hi - lo) / Na
a_values = []
f = Constant((0.0, -1.0))

g_tau = Constant((0.0, 0.0))  # Tangential stress component


# Viscous flux operator
def F_v(u, grad_u, p_local=None):
    if p_local is None:
        p_local = p
    return 2 * eta * sym(grad_u) - p_local * Identity(2)


for i in range(0, Na):
    print('a is ', a)
    a_values.append(a)

    # Geometry
    abnd = str(a)
    domain = Polygon([Point(0.2, 0), Point(0.2, a), Point(0.1, 1), Point(0, 1), Point(0, 0)])
    mesh = generate_mesh(domain, 50)
    n = FacetNormal(mesh)

    # Function space
    We = MixedElement([VectorElement("CG", mesh.ufl_cell(), 2),
                       FiniteElement("CG", mesh.ufl_cell(), 1)])
    W = FunctionSpace(mesh, We)
    # w = Function(W)
    # Manufactured solution

    w = Function(W)
    # weird = 'near(x[1], 1.0) && x[0]>=0.1'
    bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
    bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
    bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
    bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
    # bcu_slip = DirichletBC(W.sub(0).sub(1), Constant(0.0), weird)
    bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_symmetry]


    colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    colors.set_all(0)  # default to zero
    # We match the colours to the defined sketch in the Fenics chapter
    CompiledSubDomain("near(x[0], 0.0)").mark(colors, 5)
    CompiledSubDomain("near(x[1], 1.0) && x[0]<=0.1").mark(colors, 1)
    CompiledSubDomain("near( ( (" + abnd + "-1) /0.1)*(x[0] - 0.2) +" + abnd + "- x[1], 0.0) && x[0]>=0.1").mark(colors,
                                                                                                                 2)
    CompiledSubDomain("near(x[0], 0.2)").mark(colors, 3)  # wall
    CompiledSubDomain("near(x[1], 0.0)").mark(colors, 4)  # outflow
    # Create the measure
    ds = Measure("ds", subdomain_data=colors)

    # Variational formulation
    u, p = split(w)
    v, q = split(TestFunction(W))

    N = inner(F_v(u, grad(u)), grad(v)) * dx - dot(f, v) * dx \
        + div(u) * q * dx - dot(dot(F_v(u, grad(u), p), v), n) * ds(4) - dot(dot(F_v(u, grad(u), p), v), n) * ds(3) \
        - dot(dot(F_v(u, grad(u), p), v), n) * ds(1)
    # Slip boundary conditions
    stokes_nitsche = StokesNitscheBoundary(F_v, u, p, v, q)
    # N += stokes_nitsche
    N += stokes_nitsche.slip_nitsche_bc_residual(u_soln, g_tau, ds(2))
    solve(N == 0, w, bcs)

    # Plot solutions
    (u, p) = w.split()
    #sigma_fs = TensorFunctionSpace(mesh, "CG", 1)
    #stress_tensor = project(F_v(u, grad(u), p), sigma_fs, solver_type='cg', preconditioner_type='hypre_amg')
    #surface_traction_top = project(dot(F_v(u, grad(u)), n[1]), VectorFunctionSpace(mesh, 'DG', 1))
    #surface_traction_top_normal = project(dot(dot(F_v(u, grad(u)), n[1]), n[1]), FunctionSpace(mesh, 'DG', 1))
    #stress = surface_traction_top_normal * ds(2)
    #stress = dot(dot(stress_tensor, n), n) * ds(2)
    sigma_expr = 2 * eta * grad(u) - p * Identity(len(u))

    # Compute surface traction
    T = -sigma_expr*n

    # Compute normal and tangential components
    Tn = inner(T, n) # scalar valued
    Tt = T - Tn*n # vector valued

    # Piecewise constant test functions
    scalar = FunctionSpace(mesh, "DG", 1)

    v1 = TestFunction(scalar)

    normal_stress = Function(scalar)

    Ln = (1 / FacetArea(mesh))*v1*Tn*ds(2)
    assemble(Ln, tensor=normal_stress.vector())

    stress_array.append(np.average(normal_stress.vector()))
    a = a - da

fig = plt.figure()
plt.plot(a_values, stress_array)
plt.ylabel('Stress')
plt.xlabel('a values')
plt.title('Sweep over a values, Isothermal slip')
plt.show()
values = np.asarray([a_values, stress_array])
np.savetxt("Results/avsstress.csv", values.T, delimiter='\t')

File("Results/velocitySlipIsothermal.pvd") << u
c = plot(u, title='Velocity Isothermal')
plt.colorbar(c)
plt.show()