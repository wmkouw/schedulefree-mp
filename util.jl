"""
Utility functions and objects.

Wouter Kouw
16-08-2019
"""

export Delta

import Base:*
import Statistics: mean, var
import LightGraphs: edges

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

function *(px::Normal{Float64}, qx::Normal{Float64})
    "Multiplication of two normal distributions of the same variable."

    # Extract parameters
    mu_p, si_p = params(px)
    mu_q, si_q = params(qx)

    # Add precisions
    W = inv(si_p) + inv(si_q)

    # Add precision-weighted means
    Wm = inv(si_p)*mu_p + inv(si_q)*mu_q

    # Return normal distribution in mean-variance parameterization
    return Normal(inv(W)*Wm, inv(W))
end

function *(px::Gamma{Float64}, qx::Gamma{Float64})
    "Multiplication of two gamma distributions of the same variable."

    # Extract parameters
    shape_p, scale_p = params(px)
    shape_q, scale_q = params(qx)

    # Add shapes
    shape = shape_p + shape_q

    # Add scales
    scale = inv(scale_p) + inv(scale_q)

    # Return gamma distribution in shape-scale parameterization
    return Gamma(shape-1., inv(scale))
end

function key_from_value(d::Dict{String,String}, k::String)
    "Given a value, find its key in a paired string dictionary."
    return collect(keys(d))[findfirst(collect(values(d)) .== k)]
end

function edges(g::LightGraphs.SimpleGraphs.AbstractSimpleGraph, node_id)
    "Get edges connected to given node"

    # Preallocate edges subset
    edges_ = Vector{LightGraphs.SimpleGraphs.AbstractSimpleEdge}()

    # Iterate through edges in graph
    for edge in edges(G)

        # Check whether edge is connected to target node
        if edge.src == node_id

            push!(edges_, edge)
        end
    end
    return edges_
end
