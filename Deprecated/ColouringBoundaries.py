from dolfin import *
from mshr import *
# Define mesh and geometry
a = 0.7
b = 0.8

domain = Polygon([Point(1, 0), Point(1, a), Point(b, (a+1)/2), Point(0.5, 1), Point(0, 1), Point(0, 0)])
mesh = generate_mesh(domain, 50)
n = FacetNormal(mesh)

# Define Taylor--Hood function space W 
V = VectorElement("CG", triangle, 2)
Q = FiniteElement("CG", triangle, 1)
W = FunctionSpace(mesh, MixedElement([V, Q]))

# Define Function and TestFunction(s)
w = Function(W); (u, p) = split(w)
(v, q) = split(TestFunction(W))


colors = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
colors.set_all(0) # default to zero

CompiledSubDomain("near(x[0], 0.0)").mark(colors, 5)
CompiledSubDomain("near(x[1], 1.0) && x[0]<=0.5").mark(colors, 1)
abnd = str(a)
bbnd = str(b)
CompiledSubDomain(
    "near( ( (" + abnd + "-1) /(2*( 1 -" + bbnd + ")))*(x[0] - 1) +" + abnd + "- x[1], 0.0) && x[0]>=0.5").mark(colors,
                                                                                                                2)
CompiledSubDomain("near( ( (" + abnd + "-1) /(2*(" + bbnd + "-0.5)))*(x[0] - 0.5) + 1 - x[1], 0.0) && x[0]>=0.5").mark(
    colors, 2)
CompiledSubDomain("near(x[0], 1.0)").mark(colors, 3)  # wall
CompiledSubDomain("near(x[1], 0.0)").mark(colors, 4)  # outflow
File("Results/colours.pvd") << colors