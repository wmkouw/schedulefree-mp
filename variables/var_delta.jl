export VarDelta

using Distributions: Normal, logpdf

mutable struct VarDelta
    """
    Variable clamped to an observation
    """

    # Node properties
    id::Symbol
    marginal::Delta
    time::Union{Integer,Nothing}
    block::Bool

    # Free energy fields
    free_energy::Float64
    grad_free_energy::Tuple{Float64,Float64}

    function VarDelta(id::Symbol,
					  observed_value::Float64;
					  time::Union{Integer,Nothing}=0,
					  block=false)

        # Cast observed value to Delta distrbution
        marginal = Delta(observed_value)

        return new(id, marginal, time, block)
    end
end

function entropy(variable::VarDelta)
    "An observed variable has no entropy."
    return 0.0
end

function gradient_entropy(variable::VarDelta)
    "Gradient of entropy for observed variable is 0."
    return 0.0
end

function free_energy(graph::MetaGraph, variable::VarDelta)
    "Free energy at observation node is node energy evaluated at observation."
	#TODO
    return 0.
end

function grad_free_energy(graph::MetaGraph, variable::VarDelta)
    "Free energy at observation node is node energy evaluated at observation."
	#TODO
    return Inf, Inf
end

function prediction_error(variable::VarDelta, message)
    "Compute precision-weighted prediction error"
    return -logpdf(message, variable.observation)
end

function belief(variable::VarDelta)
    "Edge is distribution"
    return Delta(variable.observation)
end

function act!(graph::MetaGraph, variable::VarDelta)
    "Pass message to edge with likelihood node"

    # Check each connected edge
    for edge in edges(graph, graph[variable.id, :id])

        # Check for incoming
		set_prop!(graph, edge, :message_var2factor, variable.marginal)
		set_prop!(graph, edge, :∇free_energy, norm(variable.grad_free_energy))
    end
end

function react!(graph::MetaGraph, variable::VarDelta)
    "React to incoming messages"

    # Compute local free energy
    variable.free_energy = free_energy(graph, variable)

    # Compute gradient of free energy
    variable.grad_free_energy = grad_free_energy(graph, variable)

    # Message from edge to nodes
    act!(graph, variable)
end
