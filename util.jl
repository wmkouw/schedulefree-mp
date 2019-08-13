"""
Set of utility functions.
"""

function mean(m::Float64)
    """Mean of a scalar is the scalar"""
    return m
end

function key_from_value(d::Dict{String,String}, k::String)
    "Given a value, find its key in a paired string dictionary."
    return collect(keys(d))[findfirst(collect(values(d)) .== k)]
end
