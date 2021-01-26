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
