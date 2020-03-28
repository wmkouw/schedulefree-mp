"""
Utility functions

Wouter Kouw
28-03-2020
"""


function key_from_value(d::Dict{String, Union{Float64, Symbol}}, k::Union{String, Symbol})
    "Given a value, find its key in a paired string dictionary."
    return collect(keys(d))[findfirst(collect(values(d)) .== k)]
end
