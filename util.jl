"""
Utility functions and objects.

Wouter Kouw
16-08-2019
"""

export Delta

import Base: *
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

function moments(d::Delta)
    return mean(d), var(d)
end

function moments(p::Normal)
    "First two moments of a Normal distribution"
    return mean(p), var(p)
end

function moments(p::Gamma)
    "First two moments of a Gamma distribution"
    return mean(p), var(p)
end

function *(px::Normal{Float64}, qx::Normal{Float64})
    "Multiplication of two normal distributions of the same variable."

    # Extract parameters
    μ_p, σ_p = params(px)
    μ_q, σ_q = params(qx)

    # Compute precisions
    τ_p = inv(σ_p^2)
    τ_q = inv(σ_q^2)

    # Add precisions
    τ = τ_p + τ_q

    # Add precision-weighted means
    τμ = τ_p*μ_p + τ_q*μ_q

    # Compute new variance
    σ2 = inv(τ)

    # Return normal distribution in mean-stddev parameterization
    return Normal(σ2*τμ, sqrt(σ2))
end

function *(px::Gamma{Float64}, qx::Gamma{Float64})
    "Multiplication of two Gamma distributions of the same variable."

    # Extract parameters
    shape_p, scale_p = params(px)
    shape_q, scale_q = params(qx)

    # Add shapes
    new_shape = shape_p + shape_q

    # Add inverse scales
    new_scale = inv(inv(scale_p) + inv(scale_q))

    # Return gamma distribution in shape-scale parameterization
    return Gamma(new_shape - 1., new_scale)
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
