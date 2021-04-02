from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys, os


def main():
    ########################
    # Problem parameters
    ########################
    rIn  = 0.75				# radius of pore (top, inlet)
    rOut = 1.25				# radius of pore (bottom, outlet)
    uIn  = 1				# max velocity at inlet
    l    = 5				# length of pore
    Pe   = 10			    # Peclet number
    a1 	 = 0.4				# reaction rate (deposit on surface)
    a2 	 = 0.8    			# thickness growth rate
    cIn  = 1                # concentration of pollutant at inlet

    r1 = 1.25
    r2 = 0.75
    Pe = [10, 1000]
    for pe in Pe:
        for rIn, rOut in zip([r1, 1, r2], [r2, 1, r1]):
            Pore_Clogging(rIn, rOut, uIn, l, pe, a1, a2, cIn)
    return


def Pore_Clogging(rIn, rOut, uIn, l, Pe, a1, a2, cIn):
    # discretisation details
    TF   = 100              # final time
    NT   = 5000             # number of time steps
    dt   = TF/NT            # timestep size
    Nx 	 = 64               # mesh nodes
    Ny 	 = 32

    ########################
    # Define function spaces
    ########################
    mesh = UnitSquareMesh(Nx,Ny)
    vEl = VectorElement("Lagrange", mesh.ufl_cell(), 2) # for velocity
    qEl = FiniteElement("Lagrange", mesh.ufl_cell(), 1) # for pressure
    W = FunctionSpace(mesh, vEl*qEl)

    P = FunctionSpace(mesh, "Lagrange", 1 )              # for pressure
    S = FunctionSpace(mesh, "Lagrange", 2 )              # for concentration
    V = VectorFunctionSpace(mesh, "Lagrange", 2, dim=2 ) # for velocity
    T = TensorFunctionSpace(mesh, "Lagrange", 2)         # for map's Jacobian

    # Define test and trial functions
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    c = TrialFunction(S)
    s = TestFunction(S)

    # Functions for solving problems
    _w = Function(W)
    _u, _p = split(_w)
    _c = Function(S)
    _c0 = Function(S)

    h = Function(S)
    dh = Function(S)


    ###################### 
    # Defining expressions
    ###################### 
    u0 = Expression( ('A * ( 1.0 - pow(x[1],2) )', '0.0' ), degree=2, A=Constant(uIn) )
    x0 = Expression('x[0]', degree=1)
    x1 = Expression('x[1]', degree=1)

    cppfile = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + 'expressions.cpp'
    with open(cppfile, 'r') as f:
        cpp_code=f.read()
    cpp_obj = compile_cpp_code(cpp_code)

    # Reactive boundary shape
    h.assign( interpolate(Expression( 'B + x[0]*(A-B)', B=Constant(rIn), A=Constant(rOut), degree=2 ), S ) )
    # h.assign(interpolate(Expression( 'B + x[0]*(A-B)+0.1*sin(2*pi*x[0]*3)', B=Constant(rIn), A=Constant(rOut), degree=2 ), S ))
    # h = interpolate(Expression( 'B + x[0]*(A-B)+3*x[0]*(1-x[0])', B=Constant(rIn), A=Constant(rOut), degree=2 ), S )

    # Concentration at free boundary 
    bdrUpdate = CompiledExpression(cpp_obj.FreeBdrUpdate(_c0.cpp_object()), degree=2, domain=mesh)

    # Concentration at the outlet
    outletConc = CompiledExpression(cpp_obj.OutletConc(_c.cpp_object()), degree=2, domain=mesh)

    # define useful stuff for mapping
    J = CompiledExpression(cpp_obj.Jacobian(l, h.cpp_object(), dh.cpp_object()), degree=2, domain=mesh)
    Jinv  = inv(J)
    JinvT = inv(J.T)
    M = Jinv * JinvT
    dh.assign( project(Dx(h,0), S) )



    ##################
    # Boundary marking
    ##################

    class BdInflow(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0.0)

    class BdOutflow(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 1.0)


# Reactive boundary shape
#h.assign(interpolate(Expression( 'B + x[0]*(A-B)+0.1*sin(2*pi*x[0]*3)', B=Constant(rIn), A=Constant(rOut), degree=2 ), S ))
    h = interpolate(Expression( 'B + x[0]*(A-B)+3*x[0]*(1-x[0])', B=Constant(rIn), A=Constant(rOut), degree=2 ), S )

    class BdFree(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 1.0)


    class BdSym(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0.0)

    bdInflow  = BdInflow()
    bdoutflow = BdOutflow()
    bdSym 	  = BdSym()
    bdFree 	  = BdFree()

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundaries.set_all(0)

    bdInflow.mark(boundaries, 1)
    bdoutflow.mark(boundaries, 2)
    bdSym.mark(boundaries, 3)
    bdFree.mark(boundaries, 4)

    dss = ds(domain=mesh, subdomain_data=boundaries)

    # Boundary conditions
    dirBCs = [ DirichletBC( W.sub(0), u0, boundaries, 1 ), \
            DirichletBC( W.sub(0), Constant((0.0, 0.0)), boundaries, 4 ),
            DirichletBC( W.sub(0).sub(1), Constant(0.0), boundaries, 3 )]

    dirBCc = [ DirichletBC( S, Constant(cIn), boundaries, 1 ) ]



    ###################
    # Variational forms
    ###################

    # To recover a reasonable inflow distribution for the concentration, first solve on the square domain
    # solve Stokes
    aStokes0 = inner( grad(u), grad(v) ) * x1 * dx \
            - p * div( v )              * x1 * dx \
            - p * v[1]                       * dx \
            + q * div(u)                * x1 * dx \
            + q * u[1]                       * dx      # symmetry BC
    Ls       = Constant(0) * q           * x1 * dx

    solve( aStokes0 == Ls, _w, dirBCs )

    # solve concentration
    aAdvDif0 = inner( grad(c), grad(s) )     * x1 * dx \
            + Pe * inner( grad(c), _u ) * s * x1 * dx \
            + a1 * c                    * s * x1 * dss(4)
    Lad0     = Constant(0)               * s * x1 * dx
        
    solve( aAdvDif0 == Lad0, _c, dirBCc )

    # use the outlet profile (rescaled) as inlet profile for the next simulations
    cin = project( outletConc / assemble(_c*dss(2)) * cIn, S )
    dirBCc = [ DirichletBC( S, cin, boundaries, 1 ) ]



    # assemble weak formulations
    # Stokes
    aStokes = inner( M*grad(u), grad(v) ) * x1 * h * abs(det(J)) * dx \
            - p * div( Jinv*v )           * x1 * h * abs(det(J)) * dx \
            - p * ( Jinv *v )[1]               * h * abs(det(J)) * dx \
            + q * tr( JinvT*grad(u) )     * x1 * h * abs(det(J)) * dx \
            + q * ( JinvT*u )[1]               * h * abs(det(J)) * dx
    Ls      = Constant(0) * q             * x1 * h * abs(det(J)) * dx

    # solve Stokes
    solve( aStokes == Ls, _w, dirBCs )

    # concentration
    aAdvDif = Constant(dt) * inner( M*grad(c), grad(s) )         * x1 * h * abs(det(J)) * dx \
            + Constant(dt) * Pe * inner( JinvT*grad(c), _u ) * s * x1 * h * abs(det(J)) * dx \
            + Constant(dt) * a1 * c                          * s * x1 * h * abs(det(J)) * dss(4)

    Lad     = Constant(0)                                    * s                        * dx

    # solve concentration for initial guess
    solve( aAdvDif == Lad, _c, dirBCc )
    _c0.vector()[:] = _c.vector()


    # Adding time derivative
    aAdvDif += c    * s * x1 * h * abs(det(J)) * dx
    Lad     += _c0  * s * x1 * h * abs(det(J)) * dx



    ##############################
    # Mapping and saving solutions
    ##############################

    # Define function for mapping domains
    def F(x):
        result = np.zeros((len(x[:,0]),2))
        for i in range(len(x[:,0])):
            result[i,:] = ( x[i,0]*l, x[i,1]*h(x[i,0],x[i,1]) )
        return result

    # New spaces on the mapped domain
    mappedMesh = Mesh(mesh)
    Sm   = FunctionSpace(mappedMesh, "Lagrange", 2 )           # for concentration
    Velm = VectorElement("Lagrange", mappedMesh.ufl_cell(), 2) # for velocity
    Qelm = FiniteElement("Lagrange", mappedMesh.ufl_cell(), 1) # for pressure
    Wm = FunctionSpace(mappedMesh, Velm*Qelm)
    wm = Function(Wm)
    um, pm = wm.split(True)
    cm = Function(Sm)

    # Creating vtk files 
    folder = "Pe_"+str(Pe)+"_Rin_"+str(rIn)+"_Rout_"+str(rOut)
    ufile = File(folder+"/velocity.pvd")
    pfile = File(folder+"/pressure.pvd")
    cfile = File(folder+"/concentration.pvd")
    um.rename('u', 'velocity')
    pm.rename('p', 'pressure')
    cm.rename('c', 'concentration')

    # Mapping and saving solutions  
    def MapAndSave(mappedMesh, newCoords, mappedFuncs, funcs, files, J, V, P):
        mappedMesh.coordinates()[:] = newCoords
        mappedFuncs[0].vector()[:] = project( J * funcs[0] / abs(det(J)), V ).vector() # velocity
        mappedFuncs[1].vector()[:] = project( funcs[1], P ).vector()                   # pressure
        mappedFuncs[2].vector()[:] = funcs[2].vector()                                 # concentration
        for i,f in enumerate(files):
            f << mappedFuncs[i]
        return

    x = mesh.coordinates()
    MapAndSave(mappedMesh, F(x), [um, pm, cm], [_u, _p, _c], [ufile, pfile, cfile], J, V, P)




    ####################
    # Time-stepping loop (FE for h coz I cant be bothered)
    ####################

    for i in range( NT ):
        print("Time", dt*i)

        # update h
        h.assign( project( h - Constant(a2*dt) * bdrUpdate, S ) )
        dh.assign( project(Dx(h,0), S) )
        
        # !!! This check is wrong... Needs to be changed
        if any(t < 0 for t in h.vector().get_local()):
            print('Pore clogged!')
            break
        
        # solve Stokes
        solve( aStokes == Ls, _w, dirBCs )
        
        # solve concentration
        solve( aAdvDif == Lad, _c, dirBCc )
        _c0.vector()[:] = _c.vector()

        # map mesh
        MapAndSave(mappedMesh, F(x), [um, pm, cm], [_u, _p, _c], [ufile, pfile, cfile], J, V, P)
        pass
    return


if __name__ == "__main__":
    main()
