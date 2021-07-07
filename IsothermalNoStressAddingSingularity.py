from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
# Define mesh and geometry - We solve for half of the domain we need, and impose symmetry
#domain = Polygon([Point(1, 0), Point(1, 0.5), Point(0.5, 1), Point(0, 1), Point(0, 0)])
#mesh = generate_mesh(domain, 100)
#mesh = RectangleMesh(Point(0, 0), Point(0.2, 1.0), 80, 80, "crossed")
mesh = Mesh('Meshes/IsothermalAdaptiveMesh.xml')
n = FacetNormal(mesh)

# V = VectorElement("CG", mesh.ufl_cell(),  2)
# Q = FiniteElement("DG", mesh.ufl_cell(), 0)
# P2 = FiniteElement("CG", mesh.ufl_cell(), 2)
# B = FiniteElement("Bubble", mesh.ufl_cell(), mesh.topology().dim() + 1)
# V = VectorElement(NodalEnrichedElement(P2, B))
# Q = FiniteElement("DG", mesh.ufl_cell(), 1)

# Define Taylor--Hood function space W
V = VectorElement("CG", triangle, 2) # original spaces
Q = FiniteElement("CG", triangle, 1) # original spaces
W = FunctionSpace(mesh, MixedElement([V, Q]))

# Define Function and TestFunction(s)
w = Function(W)
(u, p) = split(w)
(v, q) = split(TestFunction(W))

# Define the viscosity and bcs

mu = Constant(1.0)
# mu = Expression('exp(-a*pow(x[0],2))', degree=2, a=10)

u_in = Constant(-2.0)
u_c = Constant(-1.0)


# Note, x[0] is r and x[1] is x, and x[1] == 0 is the bottom.
inflow = 'near(x[1], 1.0) && x[0]<=0.1' # This is C++ code.
right = 'near(x[1], 1.0) && x[0]>=0.1'
wall = 'near(x[0], 0.2)'
centre = 'near(x[0], 0.0)'
outflow = 'near(x[1], 0.0)'
pinpoint = 'near(x[1], 1.0) && near(x[0], 0.1)'
#bcp_pinpoint = DirichletBC(W.sub(1), Constant(0.0), pinpoint, "pointwise")
bcu_inflow = DirichletBC(W.sub(0), (0.0, u_in), inflow)
bcu_wall = DirichletBC(W.sub(0), (0.0, u_c), wall)
bcu_outflow = DirichletBC(W.sub(0), (0.0, u_c), outflow)

bcu_symmetry = DirichletBC(W.sub(0).sub(0), Constant(0.0), centre)
bcs = [bcu_inflow, bcu_wall, bcu_outflow, bcu_symmetry]
# Define stress tensor
# epsilon = sym(grad(u))
x = SpatialCoordinate(mesh)


def epsilon(v):
    return sym(as_tensor([[v[0].dx(0), v[0].dx(1)],
                          [v[1].dx(0), v[1].dx(1)]]))


# stress tensor
def sigma(v, p):
    return 2*mu*epsilon(v)-p*Identity(2)


# Define the variational form
f = Constant((0, -1))


facet_f = MeshFunction("size_t", mesh, mesh.topology().dim() - 1) # FACET function
#colors.set_all(0)  # default to zero
#We match the colours to the defined sketch in the Fenics chapter
CompiledSubDomain('near(x[1], 1.0) && x[0]<=0.1').mark(facet_f, 0)
CompiledSubDomain('near(x[1], 1.0) && x[0]>=0.1').mark(facet_f, 1)
CompiledSubDomain('near(x[0], 0.2)').mark(facet_f, 2)  # wall
CompiledSubDomain('near(x[1], 0.0)').mark(facet_f, 3)  # outflow
CompiledSubDomain('near(x[0], 0.0)').mark(facet_f, 4)
# Create the measure
ds = Measure("ds", domain=mesh, subdomain_data=facet_f)
#dS = Measure("dS", domain=mesh, subdomain_data=subdomains)
a1 = (inner(sigma(u, p), epsilon(v))) * dx
a2 = (- div(u) * q - dot(f, v)) * dx
b1 = - dot(dot(sigma(u, p), n), v)*ds(0) - dot(dot(sigma(u, p), n), v)*ds(2) - dot(dot(sigma(u, p), n), v)*ds(3)

F = a1 + a2 + b1
# Solve problem
solve(F == 0, w, bcs)

# Plot solutions
(u, p) = w.split()

N = 5000 # size of the arrays below
x0val_right = np.linspace(0.1, 0.2, N)
x0val_left = x0val_right - 0.1

Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
sig = Function(Vsig, name="Stress")
sig.assign(project(sigma(u, p), Vsig))

area0 = assemble(1.0*ds(0))
print("area at ds0 is", area0)

normal_stress_org = assemble(inner(sig*n, n)*ds(0))

print("Normal stress at boundary 0", normal_stress_org)

def pred_sing(x, C_n):
    return  -C_n*2*(0.1-x)**(-1/2)


def pred_sing_int(delta, C_n):
    return -C_n*2*2*(delta)**(1/2)


# We define the delta value to use, based on just guesswork x.x
delta_values = np.logspace(-1.1, -4, num=20) #This gives delta values up to delta=0.005.
stress_values = []
stress_values_2 = []
for i in range(len(delta_values)):
    delta = delta_values[i]
    x0val_delta = np.linspace(0, 0.1-delta, N)
    stress_top_left_calc = []
    for j in range(len(x0val_delta)):
        stress_top_left_calc.append(sig(x0val_delta[j], 1.0)[3])

    val = stress_top_left_calc[-1]
    x0val_rest = np.linspace(x0val_delta[-1], 0.1, N)
    #dx_calc = (0.1 - delta) / (N - 1)
    #dx_sing = (x0val_rest[-1] - x0val_rest[0]) / (N - 1)
    #C_n = np.abs(val / pred_sing(x0val_rest[0], 1.0))
    C_n = 8.04
    pred_sing_val = pred_sing(x0val_rest[0:-1], C_n)

    stress_average = np.abs(np.sum(stress_top_left_calc) * (x0val_left[-1]-x0val_left[0])) / N  + np.abs(np.sum(pred_sing_val))*(x0val_rest[-1]-x0val_rest[0]) / N
    stress_values.append(stress_average)
    stress_average = integrate.trapz(stress_top_left_calc, x0val_left) + pred_sing_int(delta, C_n)
    stress_values_2.append(stress_average)
# stress_top_left_test = []
# for i in range(len(x0val_left)):
#     stress_top_left_test.append(sig(x0val_left[i], 1.0)[3])


values = np.asarray([delta_values, stress_values])
np.savetxt("Results/DeltavsAvgStressCase1.csv", values.T, delimiter='\t')

values = np.asarray([delta_values, stress_values_2])
np.savetxt("Results/DeltavsIntStressCase1.csv", values.T, delimiter='\t')







#flux = dot(u, n) * dot(u, n) * ds(1)