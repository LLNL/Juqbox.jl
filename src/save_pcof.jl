using FileIO

"""
    save_pcof(refFileName, pcof)

Save the parameter vector `pcof` on a JLD2 formatted file with handle `pcof`

# Arguments
- `refFileName`: String holding the name of the file.
- `pcof`: Vector of floats holding the parameters.
"""
function save_pcof(refFileName:: String, pcof:: Vector{Float64})
    save(refFileName, "pcof", pcof)
end

"""
    pcof = read_pcof(refFileName) 

Read the parameter vector `pcof` from a JLD2 formatted file

# Arguments
- `refFileName`: String holding the name of the file.
"""
function read_pcof(refFileName:: String)
    dict = load(refFileName)
    pcof = dict["pcof"]
    return pcof
end

"""
	juq2qis(params, pcof, samplerate, q_ind, fileName="ctrl.dat", node_loc="c") 

Evaluate control functions and export them into a format that is readable by Qiskit.

# Arguments
- `params:: objparams`: Struct with problem definition
- `pcof:: Array{Float64,1})`: Vector of parameter values
- `samplerate:: Float64`: Samplerate for quantum device (number of samples per ns for the IQ mixer)
- `q_ind:: Int64`: Index of the control function to output (`1 <= q_ind <= Nctrl`)
- `fileName:: String`: Name of output file containing controls to be handled by Qiskit
- `node_loc:: String`: Node location, "c" for cell centered, "n" for node-centered, default is cell-centered
"""
function juq2qis(params::objparams, pcof:: Array{Float64, 1}, samplerate:: Float64, q_ind:: Int64, fileName:: String="ctrl.dat", node_loc:: String="c")

    qiskit_dt = 1/samplerate
    @assert(q_ind >=1)
    
    if params.T < qiskit_dt
	throw(ArgumentError("Final simulation time shorter than qiskit_dt, please increase final simulation time.\n"))
    end

    @assert( params.Ncoupled == 0 || params.Nunc == 0) # can't have both

    Nctrl = params.Ncoupled + params.Nunc

    # Check if we have an integer number of timesteps of size qiskit_dt in our signal
    # FG: This can be a problem if the width of a B-spline is small enough as 
    # the qiskit_dt could ask for data outside of the control function definition
    # if we don't take value from left.
    nsteps = round(Int64,params.T/qiskit_dt)
    rmdr = params.T-qiskit_dt*nsteps
    println("Number of ctrl samples = ", nsteps)
    if rmdr > 1e-10
	throw(ArgumentError("Non-integral number of timesteps of size qiskit_dt, adjust final simulation time.\n"))
    end

    # Take midpoint value of signal in interval [(k-1)*qiskit_dt,k*qiskit_dt] ∀ k
    if(node_loc=="c")
        td = collect((0:nsteps-1).*qiskit_dt)    
	td .+= 0.5*qiskit_dt  # Use the ctrl value at the center of each interval
    else
        td = collect((0:nsteps-1).*qiskit_dt)     # Use the ctrl value at the beginning of each interval
    end
    D  = zeros(length(td),2)

    # output the B-splines without carrier waves
    bc_flag = params.use_bcarrier
    params.use_bcarrier = false

    # Controls (in rad/ns)
    if(q_ind <= Nctrl)
    	p,q = Juqbox.evalctrl(params, pcof, td, q_ind) # evalctrl uses a 1-based indexing of q_ind
    	D[:,1] = copy(p)
    	D[:,2] = copy(q)
    	# D .*= 2.0
        println("Making a plot of the ctrl functions")
        plc = plot(td,p,lab="p(t)")
        plot!(td,q,lab="q(t)")
    end

    # reset the bcarrier flag
    params.use_bcarrier = bc_flag 

    # Save signal to delimited file
    if length(fileName)>0
    	open(fileName, "w") do io
	    writedlm(io, D)
	end
        println("Saved control signal in Qiskit compatible format to delimited file '", fileName, "'");
    end

    return plc
end
