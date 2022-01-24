"""ConvergenceIsothermalManufactured

This script performs a convergence test for the coupled problem where we compare it to an exact manufactured solution.
Since the mesh changes quadratically at each iteration, it is required to redefine the problem and
boundary conditions at each time.

The results are saved and plotted externally. We expect a quadratic convergence for
the velocity problem, and temperature problem, as well as linear convergence for
the pressure problem. These results hold (with temperature behaving better than estimated)

This script requires that fenicsproject be installed within the Python
environment you are running this script in, as well as numpy and matplotlib.

"""

from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
from ufl import replace

# Define mesh and geometry - We solve for half of the domain we need, and impose symmetry
mesh = RectangleMesh(Point(0, 0), Point(0.2, 1.0), 3, 3)

# Define normal vector
n = FacetNormal(mesh)

# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2)
Q = FiniteElement("CG", triangle, 1)
W = FunctionSpace(mesh, MixedElement([V, Q]))

# Define Function and TestFunction(s)
w = Function(W)
(u, p) = split(w)
(v, q) = split(TestFunction(W))

# Define the exact forcing function
f = Expression(('0.0', '2*(x[0]-0.2)'), degree=2)
f_e = project(f, W2)

# Define the exact normal stress
g = Expression('2*(1-x[1])*(x[0]-0.2)', degree=2)
W3 = FunctionSpace(mesh, Q)
g_e = project(g, W3) # p is g_e3

# Define the exact pressure
p_e = project(Expression('0.0', degree=2), W3)

# Define the viscosity and bcs
mu = Constant(1.0)

# Define the exact velocities
u_in = Expression(('0.0', '-(x[0]-0.2)'), degree=2)
u_c = Expression(('0.0', '0.0'), degree=2)


# Define the boundaries
inflow = 'near(x[1], 1.0) && x[0]<=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'

# Define the boundary conditions
bcu_inflow = DirichletBC(W.sub(0), u_c, inflow)
bcu_wall = DirichletBC(W.sub(0), u_c, wall)
bcu_outflow = DirichletBC(W.sub(0), u_in, outflow)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_symmetry]


# Define the symmetric gradient
def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1), 0],
                          [v[1].dx(0), v[1].dx(1), 0],
                          [0, 0, 0]]))


# Define the stress tensor
def sigma(v, p):
    return 2*mu*epsilon(v)-Id(p)


# Define the pressure matrix
def Id(p):
    return as_tensor([[p, 0, 0],
                      [0, p, 0],
                     [0, 0, p]])


# Define the symmetric gradient to be used on the boundary
def cond(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


# Define the stress tensor to be used on the boundary
def sigmabc(v, p):
    return 2*mu*cond(v) - p*Identity(2)

# Define the facets
facet_f = MeshFunction("size_t", mesh, mesh.topology().dim() - 1) # FACET function
#We match the colours to the defined sketch in the Fenics chapter

# Define the subdomains for the integration measures
CompiledSubDomain('near(x[1], 1.0) && x[0]<=0.1').mark(facet_f, 0)
CompiledSubDomain('near(x[1], 1.0) && x[0]>=0.1').mark(facet_f, 1)
CompiledSubDomain('near(x[0], 0.2)').mark(facet_f, 2)  # wall
CompiledSubDomain('near(x[1], 0.0)').mark(facet_f, 3)  # outflow
CompiledSubDomain('near(x[0], 0.0)').mark(facet_f, 4)
CompiledSubDomain('near(x[1], 1.0) && x[0]>0.1 && x[0]<0.2').mark(facet_f, 5)

x = SpatialCoordinate(mesh)

# Create the measure
ds = Measure("ds", domain=mesh, subdomain_data=facet_f)

# Define the variational problem
a1 = (inner(sigma(u, p), epsilon(v))) * dx
a2 = (- div(u) * q - dot(f, v)) * dx

F = a1 + a2
# Solve the variational problem
solve(F == 0, w, bcs)

# Plot solutions
(u, p) = w.split()
h_prev = mesh.hmin() # mesh size
hvalues = [h_prev] # vector for mesh sizes

# The pressure will have non-integrable corner singularities, so we use the average pressure for error calculation

p_avg_old = assemble(p * dx) / assemble(1.0 * dx(domain=mesh))

# initialise the arrays for errors
errors_u = []
errors_p = []
errors_p2 = []

# Beginning the meshing reduction procedure
for i in range(6):
    # Create the measure
    mesh = refine(mesh)
    n = FacetNormal(mesh)
    hvalues.append(mesh.hmin())
    V2 = FunctionSpace(mesh, V)
    Q2 = FunctionSpace(mesh, Q)
    # We interpolate the exact solutions onto the new mesh to impose at boundaries

    u_prev = interpolate(u, V2) # interpolating u on mesh i + 1
    p_prev = interpolate(p, Q2)

    # We repeat the procedure from above, but not onto a new mesh.

    V = VectorElement("CG", triangle, 2)
    Q = FiniteElement("CG", triangle, 1)
    W = FunctionSpace(mesh, MixedElement([V, Q]))

    w = Function(W)

    (u, p) = split(w)
    (v, q) = split(TestFunction(W))

    bcu_inflow = DirichletBC(W.sub(0), u_c, inflow)
    bcu_wall = DirichletBC(W.sub(0), u_c, wall)
    bcu_outflow = DirichletBC(W.sub(0), u_in, outflow)
    bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
    bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_symmetry]

    x = SpatialCoordinate(mesh)

    ds = Measure("ds", domain=mesh, subdomain_data=facet_f)
    a1 = (inner(sigma(u, p), epsilon(v))) * dx
    a2 = (- div(u) * q + g*q - dot(f, v)) * dx
    F = a1 + a2
    # Solve problem
    solve(F == 0, w, bcs)
    # Plot solutions
    (u, p) = w.split()

    # Save errors
    errors_u.append(errornorm(u, u_prev, norm_type='L2'))
    errors_p.append(np.abs(errornorm(p, p_prev, norm_type='L2')))


# Calculate rates of convergence for each case and save data

rvalues = [0]
for i in range(len(errors_u)-1):
    rvalues.append(np.log(errors_u[i+1]/errors_u[i])/np.log(hvalues[i+1]/hvalues[i]))
print(rvalues)

values = np.asarray([hvalues, errors_u, rvalues])
np.savetxt("Results/ErrorsVelocityIsothermal.csv", values.T, delimiter='\t')

rvalues = [0]
for i in range(len(errors_p)-1):
    rvalues.append(np.log(errors_p[i+1]/errors_p[i])/np.log(hvalues[i+1]/hvalues[i]))
print(rvalues)

values = np.asarray([hvalues, errors_p, rvalues])
np.savetxt("Results/ErrorsPressureIsothermal.csv", values.T, delimiter='\t')

values = np.asarray([hvalues, errors_p2, rvalues])
np.savetxt("Results/ErrorsPressure2Isothermal.csv", values.T, delimiter='\t')
