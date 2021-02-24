from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

# We compute, as a proof of concept. Isothermal problem, although we solve for temperature as well, with Gamma = 0.
# We approximate the free boundary with a straight line, and adjust the position of that straight line to minimize the
# resulting flux. This minimization still ends up giving something enormous, but the question is, does a minimum even
# exist? To prove that it is a minimum, I save the resulting fluxes in a list, and then plot the list [ once I figure
# out how to do this in Python ...


lo = 0.5
hi = 1
tol = 0.0001
count = 0
N = 50
a = 1

L = 5.0
R = 0.5
asp = R / L

u_in = Constant(-4.0)
u_c = Constant(-1.0)




# symmetric gradient
def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), 0, v[0].dx(1)],
                          [0, v[0] / x[0], 0],
                          [v[1].dx(0), 0, v[1].dx(1)]]))


# stress tensor
def sigma(v, p):
    return 2 * mu * epsilon(v) - p * Identity(3)


def cond(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


def sigmabc(v, p):
    return 2 * mu * cond(v) - p * Identity(2)


def div_cyl(v):
    return (1 / x[0]) * (x[0] * v[0]).dx(0) + v[1].dx(1)


total_flux_old = 0.0
flux_array = []

# while count < N or (hi-lo) < tol:
    # a = (lo + hi) / 2
Na = 200
da = (hi-lo)/Na
a_values = []
for i in range(0, Na):
    print('a is ', a)
    a_values.append(a)
    domain = Polygon([Point(1, 0), Point(1, a), Point(0.5, 1), Point(0, 1), Point(0, 0)])
    mesh = generate_mesh(domain, 100)

    # Create mesh
    n = FacetNormal(mesh)

    # Define Taylor--Hood function space W
    V = VectorElement("CG", triangle, 2)
    Q1 = FiniteElement("CG", triangle, 1)
    W = FunctionSpace(mesh, MixedElement([V, Q1]))

    # Define Function and TestFunction(s)
    w = Function(W)
    (u, p) = split(w)
    (v, q1) = split(TestFunction(W))
    # Define the viscosity and bcs

    mu = Constant(1.0)

    # Note, x[0] is r and x[1] is x, and x[1] == 0 is the bottom.
    inflow = 'near(x[1], 1.0) && x[0]<=0.5'
    wall = 'near(x[0], 1.0)'
    centre = 'near(x[0], 0.0)'
    outflow = 'near(x[1], 0.0)'
    bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
    bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
    bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
    bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
    bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_symmetry]
    # Define the variational form
    vsc = as_vector([v[0], asp*v[1]])
    usc = as_vector([u[0], asp*u[1]])
    f = Constant((0, -1))

    colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    colors.set_all(0)  # default to zero
    # We match the colours to the defined sketch in the Fenics chapter
    CompiledSubDomain("near(x[0], 0.0)").mark(colors, 5)
    CompiledSubDomain("near(x[1], 1.0) && x[0]<=0.5").mark(colors, 1)
    abnd = str(a)
    CompiledSubDomain("near( ( ("+abnd+"-1) /0.5)*(x[0] - 1) +" + abnd + "- x[1], 0.0) && x[0]>=0.5").mark(colors, 2)
    CompiledSubDomain("near(x[0], 1.0)").mark(colors, 3)  # wall
    CompiledSubDomain("near(x[1], 0.0)").mark(colors, 4)  # outflow

    x = SpatialCoordinate(mesh)

    # Create the measure
    ds = Measure("ds", subdomain_data=colors)

    a1 = (inner(sigma(usc, p), epsilon(vsc))) * x[0] * dx
    a2 = (- div_cyl(usc) * q1 - dot(f, vsc)) * x[0] * dx
    b1 = - dot(dot(sigmabc(usc, p), vsc), n) * x[0] * ds(1)
    b3 = - (1/asp) * dot(dot(sigmabc(usc, p), vsc), n) * x[0] * ds(3)
    b4 = - dot(dot(sigmabc(usc, p), vsc), n) * x[0] * ds(4)
    F = a1 + a2 + b1 + b3 + b4

    solve(F == 0, w, bcs)
    # Extract solution
    (u, p) = w.split()

    # Extract flux
    flux = dot(u, n) * dot(u, n) * ds(2)
    total_flux_new = assemble(flux)
    flux_array.append(total_flux_new)

    print("Total flux on boundary 2 is", total_flux_new)
    # if total_flux_new < total_flux_old:
    #     hi = a
    # else:
    #     lo = a
    #
    total_flux_old = total_flux_new
    count = count + 1
    a = a - da

# W2 = FunctionSpace(mesh, Q2)
# Pmu = project(mu, W2)
#
# File("Results/velocityIsothermal.pvd") << u
# c = plot(u, title='Velocity Isothermal')
# plt.colorbar(c)
# plt.show()
#
# c = plot(T, title='Temperature, uin = 2.5')
# plt.colorbar(c)
# plt.show()

# fig = plt.figure()
# plt.plot(a_values, flux_array)
# plt.ylabel('Flux')
# plt.xlabel('a values')
# plt.title('Sweep over a values, straight line free surface estimation')
# plt.show()

values = np.asarray([a_values, flux_array])
np.savetxt("Results/avsfluxisothermaluin4ref.csv", values.T, delimiter='\t')
