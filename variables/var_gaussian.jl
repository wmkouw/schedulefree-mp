export VarGaussian

import LinearAlgebra: norm
import Distributions: Normal, mean, std, var, params
include(joinpath(@__DIR__, "../util.jl"))

mutable struct VarGaussian
    """Variable with a Gaussian recognition distribution"""

    # Factor graph properties
    id::Symbol
    time::Union{Integer,Nothing}
    block::Bool

    # Recognition distribution parameters
    marginal::Normal{Float64}
    free_energy::Float64
    grad_free_energy::Tuple{Float64,Float64}

    function VarGaussian(id::Symbol;
                         marginal::Normal{Float64}=Normal(0,1e12),
                         grad_free_energy::Tuple{Float64,Float64}=(1e12,1e12),
                         free_energy::Float64=1e12,
                         time::Union{Integer,Nothing}=0,
                         block::Bool=false)

        # Construct instance
        return new(id, time, block, marginal, free_energy, grad_free_energy)
    end
end

function params(variable::VarGaussian)
    "Parameters of current belief"
    return params(variable.marginal)
end

function mean(variable::VarGaussian)
    "Mean of current belief"
    return mean(variable.marginal)
end

function var(variable::VarGaussian)
    "Variance of current belief"
    return var(variable.marginal)
end

function moments(variable::VarGaussian)
    "First two moments of current belief"
    return mean(variable), var(variable)
end

function update!(graph::AbstractMetaGraph, variable::VarGaussian)
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

function marginal(variable::VarGaussian)
    "Marginal over variable"
    return variable.marginal
end

function entropy(variable::VarGaussian)
    "Entropy of Gaussian distribution"
    return 1/2*log(2*π*sqrt(var(variable.marginal)))
end

function grad_entropy(variable::VarGaussian)
    "Gradient of entropy of Gaussian evaluated for supplied parameters"

	# Extract std dev
	σ = sqrt(var(variable.marginal))

    # Partial derivative with respect to mean
    partial_μ = 0.0

    # Partial derivative with respect to variance
    partial_σ = 1/σ

    # Return tuple of partial derivatives
    return (partial_μ, partial_σ)
end

function free_energy(graph::MetaGraph, variable::VarGaussian)
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

function grad_free_energy(graph::MetaGraph, variable::VarGaussian)
    "Compute gradient of free energy w.r.t. parameters of variable."

	# Initialize energy
	partial_μ = 0.
	partial_σ = 0.

	# Iterate over edges connected to variable node
	for edge in edges(graph, graph[variable.id, :id])

		# Extract message from factor to variable
		message_f2v = get_prop(graph, edge, :message_factor2var)

		# Compute the gradient of expectation E_qx [log p(x)] w.r.t. qx params
		partials = grad_expectation(variable.marginal, message_f2v)

		# Update partial parameters
		partial_μ += partials[1]
		partial_σ += partials[2]
	end

	# Gradient of entropy
	partials = grad_entropy(variable)

	# Update partial parameters
	partial_μ += partials[1]
	partial_σ += partials[2]

	return (partial_μ, partial_σ)
end

function act!(graph::MetaGraph, variable::VarGaussian)
    "Pass belief to edges."

    # Check each connected edge
    for edge in edges(graph, graph[variable.id, :id])

        # Check for incoming
		set_prop!(graph, edge, :message_var2factor, variable.marginal)
		set_prop!(graph, edge, :∇free_energy, norm(variable.grad_free_energy))
    end
end

function react!(graph::MetaGraph, variable::VarGaussian)
    "React to incoming messages"

    # Check whether variable is blocked
    if !variable.block

        # Update variational distribution
        update!(graph, variable)

        # Compute free energy after update
        variable.free_energy = free_energy(graph, variable)

        # Compute gradient of free energy after update
        variable.grad_free_energy = grad_free_energy(graph, variable)

        # Message from edge to nodes
        act!(graph, variable)
    end
end
