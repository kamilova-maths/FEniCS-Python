from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

# Define mesh and geometry - We solve for half of the domain we need, and impose symmetry
#mesh = Mesh('Meshes/IsothermalRefinedMesh.xml')
mesh = RectangleMesh(Point(0, 0), Point(0.2, 1), 100, 100)
n = FacetNormal(mesh)
lo = 0.5
hi = 1
tol = 0.0001
count = 0
a = 1


# Define stress tensor
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


# Define the variational form
f = Constant((0, -1)) # forcing term
a_values = []
stress_array = []
Na = 10
da = (hi-lo)/Na

for i in range(Na):
    print('a is' , a)
    a_values.append(a)
    domain = Polygon([Point(0.2, 0), Point(0.2, a), Point(0.1, 1), Point(0, 1), Point(0, 0)])
    mesh = generate_mesh(domain, 100)
    # Create mesh
    n = FacetNormal(mesh)
    # Define Taylor--Hood function space W
    V = VectorElement("CG", triangle, 2)
    Q = FiniteElement("CG", triangle, 1)
    W = FunctionSpace(mesh, MixedElement([V, Q]))

    # Define Function and TestFunction(s)
    w = Function(W)
    (u, p) = split(w)
    (v, q) = split(TestFunction(W))

    # Define the viscosity and bcs
    def right(x, on_boundary):
        return on_boundary and near( ( ( a -1) /0.1)*(x[0] - 0.2) + a - x[1], 0.0) and x[0]>=0.1


    mu = Constant(1.0)
    u_in = Constant(-2.0)  # inlet velocity
    u_c = Constant(-1.0)  # wall velocity, and outlet velocity
    inflow = 'near(x[1], 1.0) && x[0]<=0.1'  # left half of top boundary
    wall = 'near(x[0], 0.2)'  # right boundary
    centre = 'near(x[0], 0.0)'  # left boundary
    outflow = 'near(x[1], 0.0)'  # bottom boundary

    bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
    bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
    bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)
    bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
    bcu_slip = DirichletBC(W.sub(0).sub(1), Constant(0.0), right)  # slip condition
    bcs = [bcu_inflow, bcu_wall, bcu_symmetry, bcu_slip]

    x = SpatialCoordinate(mesh)
    colors = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    colors.set_all(0)  # default to zero
    CompiledSubDomain("near(x[0], 0.0)").mark(colors, 4)
    CompiledSubDomain("near(x[1], 1.0) && x[0]<=0.1").mark(colors, 0) # top left
    abnd = str(a)
    CompiledSubDomain("near( ( (" + abnd + "-1) /0.1)*(x[0] - 0.2) +" + abnd + "- x[1], 0.0) && x[0]>=0.1").mark(colors,
                                                                                                                 1)
    #CompiledSubDomain("near(x[1], 1.0) && x[0]>=0.1").mark(colors, 1) # top right
    CompiledSubDomain("near(x[0], 0.2)").mark(colors, 2)  # wall
    CompiledSubDomain("near(x[1], 0.0)").mark(colors, 3)  # outflow

    # Create the measure
    ds = Measure("ds", subdomain_data=colors)

    a1 = (inner(sigma(u, p), epsilon(v))) * dx
    a2 = (- div(u) * q - dot(f, v)) * dx

    F = a1 + a2

    # Solve problem
    solve(F == 0, w, bcs)

    # Plot solutions
    (u, p) = w.split()

    # File("Results/velocityIsothermalSlip.pvd") << u
    # c = plot(u, title='Velocity Isothermal')
    # plt.colorbar(c)
    # plt.show()

    #File("Results/pressureIsothermalSlip.pvd") << p
# Compute stress tensor
    sigma_expr = 2 * mu * grad(u) - p*Identity(len(u))

# Compute surface traction
    T = -sigma_expr*n

# Compute normal and tangential components
    Tn = inner(T, n)
# scalar valued
    Tt = T - Tn*n
# vector valued

    # Piecewise constant test functions
    scalar = FunctionSpace(mesh, "DG", 1)
    #vector = VectorFunctionSpace(mesh, "DG", 2)
    v1 = TestFunction(scalar)
    #w1 = TestFunction(vector)

    # Assemble piecewise constant functions for stress
    normal_stress = Function(scalar)
    #shear_stress = Function(vector)

    Ln = (1 / FacetArea(mesh))*v1*Tn*ds(1)
    #Ln = v1*Tn*ds(subdomain_data=colors, subdomain_id=1)/(Constant(1.0)*ds(subdomain_data=colors, subdomain_id=1))
    #Lt = (1 / FacetArea(mesh))*inner(w1,Tt)*ds(1)
    assemble(Ln, tensor=normal_stress.vector())
    #assemble(Lt, tensor=shear_stress.vector())
    stress_array.append(np.average(normal_stress.vector()))
    a = a - da

fig1 = plt.figure()
plt.plot(a_values, stress_array)
plt.ylabel('Stress values')
plt.xlabel('a')
plt.show()
values = np.asarray([a_values, stress_array])
np.savetxt("Results/avsstress.csv", values.T, delimiter='\t')

File("Results/velocitySlipIsothermal.pvd") << u
c = plot(u, title='Velocity Isothermal')
plt.colorbar(c)
plt.show()