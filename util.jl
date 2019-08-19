"""
Utility functions and objects.

Wouter Kouw
16-08-2019
"""

export Delta

import Statistics: mean, var

mutable struct Delta
    """
    Delta distribution.

    This is not a proper distribution, but this structure allows for a general
    interface for observed variables and unobserved variables (i.e. it is
    possible to call mean/var on a fixed variable).
    """

    # Attributes
    value::Float64

    function Delta(value::Float64)
        return new(value);
    end
end

function mean(d::Delta)
    return d.value
end

function var(d::Delta)
    return 0.0
end


function key_from_value(d::Dict{String,String}, k::String)
    "Given a value, find its key in a paired string dictionary."
    return collect(keys(d))[findfirst(collect(values(d)) .== k)]
end
