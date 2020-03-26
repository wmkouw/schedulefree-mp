"""
Utility functions and objects.

Wouter Kouw
16-08-2019
"""

export Delta

import Base: *
import Statistics: mean, var
import LightGraphs: edges, add_edge!
import MetaGraphs: set_props!
import Distributions: ContinuousUnivariateDistribution

struct Delta <: ContinuousUnivariateDistribution
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

struct Flat <: ContinuousUnivariateDistribution
    """
    Non-informative (flat) prior distribution.

    This is an improper prior that treats each element of its sample space as
    equally likely. Here, this object is used as an identity function for
    multiplication with other probability distrbutions. In other words:
    Flat(x) ⋅ Normal(x | 0,1) = Normal(x | 0,1).
    """
    Flat() = new()
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
    "First two moments of a Normal distribution."
    return mean(p), var(p)
end

function moments(p::Gamma)
    "First two moments of a Gamma distribution."
    return mean(p), var(p)
end

function moments(graph::MetaGraph, x::Float64)
    "If variable is clamped to a value, then return that value with variance 0."
    return x, 0
end

function moments(graph::MetaGraph, x::Symbol)
    "Extract marginal from variable in graph and return moments."
    return moments(graph[graph[x, :id], :node].marginal)
end

function *(a::Symbol, b::Integer)
    "Concatenate symbol and integer"
    return Symbol(string(a)*string(b))
end

function *(a::Symbol, b::String)
    "Concatenate symbol and integer"
    return Symbol(string(a)*b)
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
    new_shape = shape_p + shape_q - 1.

    # Add inverse scales
    new_scale = inv(inv(scale_p) + inv(scale_q))

    # Return gamma distribution in shape-scale parameterization
    return Gamma(new_shape, new_scale)
end

function *(px::Union{Gamma{Float64},Normal{Float64}}, qx::Flat)
    "Multiplication of a distribution with a flat distribution."
    return px
end

function *(qx::Flat, px::Union{Gamma{Float64},Normal{Float64}})
    "Multiplication of a distribution with a flat distribution."
    return px
end

function expectation(q::Normal{Float64}, ν::Normal{Float64})
    "Compute the expectation E_q(x) [log ν(x)]"

    # Parameters
    μ_q, σ_q = params(q)
    μ_ν, σ_ν = params(ν)

    return -log(σ_ν) - 1/2*log(2*π) -1/(2*σ_ν^2)*(σ_q^2 + μ_q^2 -2*μ_q*μ_ν + μ_ν^2)
end

function expectation(q::Gamma{Float64}, ν::Gamma{Float64})
    "Compute the expectation E_q(x) [log ν(x)]"

    # Parameters
    α_q, θ_q = params(q)
    α_ν, θ_ν = params(ν)

    return -log(gamma(α_ν)) - α_ν*log(θ_ν) + (α_ν-1)*(digamma(α_q) + log(θ_q)) - (α_q*θ_q)/θ_ν
end

function expectation(q::Union{Gamma{Float64},Normal{Float64}}, ν::Flat)
    "If the message is a flat distribution, the integral E_q(x) [log ν(x)] diverges."
    return 0
end

function expectation(ν::Flat, q::Union{Gamma{Float64},Normal{Float64}})
    "If the message is a flat distribution, the integral E_q(x) [log ν(x)] diverges."
    return 0
end

function grad_expectation(q::Normal{Float64}, ν::Normal{Float64})
    "Compute the gradient of the expectation E_q(x) [log ν(x)] w.r.t. q(x)'s params"

    # Parameters
    μ_q, σ_q = params(q)
    μ_ν, σ_ν = params(ν)

    # Partial derivative w.r.t μ_q
    partial_μ = -(μ_q - μ_ν)/(σ_ν^2)

    # Partial derivative w.r.t σ_q
    partial_σ = - σ_q / σ_ν

    return (partial_μ, partial_σ)
end

function grad_expectation(q::Gamma{Float64}, ν::Gamma{Float64})
    "Compute the gradient of the expectation E_q(x) [log ν(x)] w.r.t. q(x)'s params"

    # Parameters
    α_q, θ_q = params(q)
    α_ν, θ_ν = params(ν)

    # Partial derivative w.r.t α_q
    partial_α = (α_ν-1)*polygamma(1, α_q) - θ_q / θ_ν

    # Partial derivative w.r.t σ_q
    partial_θ = (α_ν-1)/θ_q - α_q / θ_ν

    return (partial_α, partial_θ)
end

function grad_expectation(q::Union{Gamma{Float64},Normal{Float64}}, ν::Flat)
    "The gradient of a diverging integral, diverges."
    return (0, 0)
end

function grad_expectation(ν::Flat, q::Union{Gamma{Float64},Normal{Float64}})
    "The gradient of a diverging integral, diverges."
    return (0, 0)
end

"""Graph-based utility functions"""

function nodes_t(graph::AbstractMetaGraph, timeslice::Integer; include_notime::Bool=false)
    "List of nodes, by :id, in current timeslice"

    # Preallocate list
    nodes_t = Any[]

    # Filter vertex by :time property
    for node in filter_vertices(graph, :time, timeslice)

        # Push to list
        push!(nodes_t, get_prop(graph, node, :id))
    end

    if include_notime

        # Find nodes without time subscript
        for node in filter_vertices(graph, :time, nothing)

            # Push to list
            push!(nodes_t, get_prop(graph, node, :id))
        end
    end

    return nodes_t
end

function edges(graph::AbstractMetaGraph, node_id::Union{Number, Symbol})
    "Get edges connected to given node"

    # Retrieve node number based on id
    if isa(node_id, Symbol)
        node_id = graph[node_id, :id]
    end

    # Preallocate edges subset
    edges_ = Vector{LightGraphs.SimpleGraphs.AbstractSimpleEdge}()

    # Iterate through edges in graph
    for edge in edges(graph)

        # Check whether edge is connected to target node
        if (edge.src == node_id) | (edge.dst == node_id)
            push!(edges_, edge)
        end
    end
    return edges_
end

function add_edge!(graph::AbstractMetaGraph,
                   edge::Tuple{Int64, Int64};
                   message_factor2var::UnivariateDistribution=Flat(),
                   message_var2factor::UnivariateDistribution=Flat(),
                   ∇free_energy::Float64=Inf)
    "Overload add_edge! to initialize message and belief dictionaries."

    if (edge[1] >= 1) & (edge[2] >= 1)

        # Connect two node
        add_edge!(graph, edge[1], edge[2])

        # Edge id
        edge_id = Edge(edge[1], edge[2])

        # Add message and belief dictionaries to edge
        set_prop!(graph, edge_id, :message_var2factor, message_var2factor)
        set_prop!(graph, edge_id, :message_factor2var, message_factor2var)
        set_prop!(graph, edge_id, :∇free_energy, ∇free_energy)
    end
end

"""Inference-based utility functions"""

function act!(graph::MetaGraph, node_num::Union{Symbol, Integer})
    "Boiler-plate forwarding"

    # Convert node symbol to node number
    if isa(node_num, Symbol)
        node_num = graph[node_num, :id]
    end

    # Make sure :node is an index
    set_indexing_prop!(graph, :node)

    # Find node based on node number
    node = graph[node_num, :node]

    # Let node act
    act!(graph::MetaGraph, node)

end

function react!(graph::MetaGraph, node_num::Union{Symbol, Integer})
    "Boiler-plate forwarding"

    # Convert node symbol to node number
    if isa(node_num, Symbol)
        node_num = graph[node_num, :id]
    end

    # Make sure :node is an index
    set_indexing_prop!(graph, :node)

    # Find node based on node number
    node = graph[node_num, :node]

    # Let node act
    react!(graph::MetaGraph, node)

end

"""Miscellaneous utility functions """

function key_from_value(d::Dict{String, Union{Float64, Symbol}}, k::Union{String, Symbol})
    "Given a value, find its key in a paired string dictionary."
    return collect(keys(d))[findfirst(collect(values(d)) .== k)]
end
