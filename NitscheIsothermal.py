# EVERYTHING HERE IS ABSOLUTELY WRONG. YOU NEED TO REWRITE THIS WHOLE THING.

from fenics import *
from dolfin import *
from dolfin_dg import StokesNitscheBoundary, tangential_proj
import matplotlib.pyplot as plt
# Geometry
mesh = UnitSquareMesh(32, 32)
n = FacetNormal(mesh)
x = SpatialCoordinate(mesh)
# Function space
We = MixedElement([VectorElement("CG", mesh.ufl_cell(), 2),
FiniteElement("CG", mesh.ufl_cell(), 1)])
W = FunctionSpace(mesh, We)


u_in = Constant(-3.5)
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
bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_symmetry, bcu_slip]
# Define stress tensor
# epsilon = sym(grad(u))
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

# Manufactured solution
u_soln = Expression(("2*x[1]*(1.0 - x[0]*x[0])","-2*x[0]*(1.0 - x[1]*x[1])"),degree=4, domain=mesh)
p_soln = Constant(0.0)
# Construct an initial guess with no singularity in eta(u)
U = interpolate(Expression(("x[1]", "x[0]", "0.0"), degree=1), W)
# Viscosity model
# def eta(u):
#     return 1 + sqrt(inner(grad(u), grad(u)))**-1

eta = Constant(1.0)


# Viscous flux operator
def F_v(u, grad_u, p_local=None):
    if p_local is None:
        p_local = p
    return 2 * eta * sym(grad_u) - p_local * Identity(2)


def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1), 0],
                          [v[1].dx(0), v[1].dx(1), 0],
                          [0, 0, 0]]))

# stress tensor
def sigma(v, p):
    return 2*eta*epsilon(v)-Id(p)


def Id(p):
    return as_tensor([[p, 0, 0],
                      [0, p, 0],
                     [0, 0, p]])


def cond(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


def sigmabc(v, p):
    return 2*eta*cond(v) - p*Identity(2)

f1 = Constant((0, -1))

# Variational formulation
u, p = split(U)
v, q = split(TestFunction(W))
#f = -div(F_v(u_soln, grad(u_soln), p_soln))
g_tau = 0*tangential_proj(F_v(u_soln, grad(u_soln), p_soln) * n, n)
N = inner(F_v(u, grad(u)), grad(v)) * dx + div(u) * q * dx + dot(f1, v) * dx
N +=  dot(dot(sigmabc(u, p), v), n) * ds(1)
N +=  dot(dot(sigmabc(u, p), v), n) * ds(3)
N +=  dot(dot(sigmabc(u, p), v), n) * ds(4)
# Slip boundary conditions
stokes_nitsche = StokesNitscheBoundary(F_v, u, p, v, q)
N += stokes_nitsche.slip_nitsche_bc_residual(u_soln, g_tau, ds(2))
solve(N == 0, U)


(u, p) = U.split()
#File("Results/velocityIsothermalFreeSlip.pvd") << u
c = plot(u, title='Velocity Isothermal')
plt.colorbar(c)
plt.show()