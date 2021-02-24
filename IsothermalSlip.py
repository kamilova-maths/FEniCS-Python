from fenics import *
from dolfin import *
from dolfin_dg import StokesNitscheBoundary, tangential_proj
import matplotlib.pyplot as plt
import numpy as np
# Geometry
mesh = RectangleMesh(Point(0, 0), Point(1, 1), 100, 100)
n = FacetNormal(mesh)
# Function space
We = MixedElement([VectorElement("CG", mesh.ufl_cell(), 2),
FiniteElement("CG", mesh.ufl_cell(), 1)])
W = FunctionSpace(mesh, We)
#w = Function(W)
# Manufactured solution
u_soln = Constant((0.0, 0.0))
#u_soln = Expression(("2*x[1]*(1.0 - x[0]*x[0])","-2*x[0]*(1.0 - x[1]*x[1])"),degree=4, domain=mesh)
p_soln = Constant(0.0)
# Construct an initial guess with no singularity in eta(u)
w = Function(W)
#U =interpolate(Expression(("x[1]", "x[0]", "0.0"), degree=1), W)
# Viscosity model

eta = Constant(1.0)
# def eta(u):
#     return 1 + sqrt(inner(grad(u), grad(u)))**-1
u_in = Constant(-2.0)
u_c = Constant(-1.0)

# Note, x[0] is r and x[1] is x, and x[1] == 0 is the bottom.
inflow = 'near(x[1], 1.0) && x[0]<=0.5'
wall = 'near(x[0], 1.0)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'
weird = 'near(x[1], 1.0) && x[0]>=0.5'
bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcu_slip = DirichletBC(W.sub(0).sub(1), Constant(0.0), weird)
bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_symmetry]

# Viscous flux operator
def F_v(u, grad_u, p_local=None):
    if p_local is None:
        p_local = p
    return 2 * eta * sym(grad_u) - p_local * Identity(2)

colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
colors.set_all(0)  # default to zero
# We match the colours to the defined sketch in the Fenics chapter
CompiledSubDomain("near(x[0], 0.0)").mark(colors, 5)
CompiledSubDomain("near(x[1], 1.0) && x[0]<=0.5").mark(colors, 1)
# CompiledSubDomain("near(x[1], x[0]*0.5/(a1-1) + 1-( 0.25/( a1-1 ) ) )", a1=a).mark(colors, 2)
CompiledSubDomain("near(x[1], 1.0) && x[0]>=0.5").mark(colors, 2)
CompiledSubDomain("near(x[0], 1.0)").mark(colors, 3)  # wall
CompiledSubDomain("near(x[1], 0.0)").mark(colors, 4)  # outflow
# Create the measure
ds = Measure("ds", subdomain_data=colors)

# Variational formulation
u, p = split(w)
v, q = split(TestFunction(W))

f = Constant((0.0, -1.0))
# f = -div(F_v(u_soln, grad(u_soln), p_soln))
g_tau = Constant((0.0, 0.0))
# g_tau = tangential_proj(F_v(u_soln, grad(u_soln), p_soln) * n, n)
N = inner(F_v(u, grad(u)), grad(v)) * dx - dot(f, v) * dx \
+ div(u) * q * dx - dot(dot(F_v(u, grad(u), p), v), n)*ds(4) - dot(dot(F_v(u, grad(u), p), v), n)*ds(3) \
    - dot(dot(F_v(u, grad(u), p), v), n)*ds(1)
# Slip boundary conditions
stokes_nitsche = StokesNitscheBoundary(F_v, u, p, v, q)
#N += stokes_nitsche
N += stokes_nitsche.slip_nitsche_bc_residual(u_soln, g_tau, ds(2))
solve(N == 0, w, bcs)

# Plot solutions
(u, p) = w.split()


File("Results/velocityIsothermalFreeSlip.pvd") << u
c = plot(u, title='Velocity Isothermal')
plt.colorbar(c)
plt.show()