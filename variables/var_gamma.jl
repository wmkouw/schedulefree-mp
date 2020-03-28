export VarGamma

import LinearAlgebra: norm
import SpecialFunctions: gamma, digamma, polygamma
include(joinpath(@__DIR__, "../prob_operations.jl"))
include(joinpath(@__DIR__, "../graph_operations.jl"))
include(joinpath(@__DIR__, "../util.jl"))

mutable struct VarGamma
    """Variable with a Gamma recognition distribution"""

    # Factor graph properties
    id::Symbol
    time::Union{Integer,Nothing}
    block::Bool

    # Recognition distribution parameters
    marginal::Gamma{Float64}
    free_energy::Float64
    grad_free_energy::Tuple{Float64,Float64}

    function VarGamma(id::Symbol;
                      marginal::Gamma{Float64}=Gamma(1.,1e-12),
                      grad_free_energy::Tuple{Float64,Float64}=(1e12,1e12),
                      free_energy::Float64=1e12,
                      time::Union{Integer,Nothing}=0,
                      block::Bool=false)

     # Construct instance
     return new(id, time, block, marginal, free_energy, grad_free_energy)
 end
end

function params(variable::VarGamma)
    "Parameters of current belief"
    return params(variable.marginal)
end

function mean(variable::VarGamma)
    "Mean of current belief"
    return mean(variable.marginal)
end

function var(variable::VarGamma)
    "Variance of current belief"
    return var(variable.marginal)
end

function moments(variable::VarGamma)
    "First two moments of current belief"
    return mean(variable), var(variable)
end

function update!(graph::MetaGraph, variable::VarGamma)
    "Update recognition distribution as the product of messages"

    # List of incoming messages
    incoming = Any[]

    # Check each connected edge
    for edge in edges(graph, graph[variable.id, :id])

		# Mesages
		message_f2v = get_prop(graph, edge, :message_factor2var)

        # Check for incoming
        push!(incoming, message_f2v)
    end

    # Update marginal
    variable.marginal = prod(incoming)
end

function marginal(variable::VarGamma)
    "Marginal over variable"
    return variable.marginal
end

function entropy(variable::VarGamma)
    "Entropy of Gamma distribution"

    # Parameters
    α = shape(variable.marginal)
    θ = scale(variable.marginal)

    # Entropy of a univariate Gamma
    return α + log(θ) + log(gamma(α)) + (1-α)*digamma(α)
end

function grad_entropy(variable::VarGamma)
    "Gradient of entropy of Gamma evaluated for supplied parameters"

	# Parameters
    α = shape(variable.marginal)
    θ = scale(variable.marginal)

    # Partial derivative with respect to shape
    partial_shape = 1 + (1 - α)*polygamma(1, α)

    # Partial derivative with respect to scale
    partial_scale = 1 / θ

    # Return tuple of partial derivatives
    return (partial_shape, partial_scale)
end

function free_energy(graph::MetaGraph, variable::VarGamma)
    "Compute local free energy of variable and connecting factor nodes"

    # Initialize energy
    U = 0.

	# Iterate over edges connected to variable node
	for edge in edges(graph, graph[variable.id, :id])

		# Extract message from factor to variable
		message_f2v = get_prop(graph, edge, :message_factor2var)

		# Compute the expectation E_qx [log ν(x)]
		U += expectation(variable.marginal, message_f2v)
	end

	# Entropy
	H = entropy(variable)

	return U - H
end

function grad_free_energy(graph::MetaGraph, variable::VarGamma)
    "Compute gradient of free energy w.r.t. recognition distribution parameters."

	# Initialize energy
	partial_α = 0.
	partial_θ = 0.

	# Iterate over edges connected to variable node
	for edge in edges(graph, graph[variable.id, :id])

		# Extract message from factor to variable
		message_f2v = get_prop(graph, edge, :message_factor2var)

		# Compute the gradient of expectation E_qx [log ν(x)] w.r.t. q(x) params
		partials = grad_expectation(variable.marginal, message_f2v)

		# Update partial parameters
		partial_α += partials[1]
		partial_θ += partials[2]
	end

	# Gradient of entropy
	partials = grad_entropy(variable)

	# Update partial parameters
	partial_α += partials[1]
	partial_θ += partials[2]

	return (partial_α, partial_θ)
end

function act!(graph::MetaGraph, variable::VarGamma)
    "Pass belief to edges."

    # Check each connected edge
    for edge in edges(graph, graph[variable.id, :id])

        # Check for incoming
		set_prop!(graph, edge, :message_var2factor, variable.marginal)
		set_prop!(graph, edge, :∇free_energy, norm(variable.grad_free_energy))
    end
end

function react!(graph::MetaGraph, variable::VarGamma)
    "React to incoming messages"

    # Check whether variable is blocked
    if !variable.block

        # Update variational distribution
        update!(graph, variable)

        # Compute free energy after update
        variable.free_energy =free_energy(graph, variable)

        # Compute gradient of free energy after update
        variable.grad_free_energy = grad_free_energy(graph, variable)

        # Message from edge to nodes
        act!(graph, variable)
    end
end
