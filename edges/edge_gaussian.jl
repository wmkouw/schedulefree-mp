export EdgeGaussian

using LinearAlgebra: inv
using Distributions: Normal, mean, std

mutable struct EdgeGaussian
    "
    Edge with a Gaussian recognition distribution
    "
    # Recognition distribution parameters
    params::Dict{String, Float64}
    change_entropy::Float64

    # Incoming messages
    message0::Normal
    message1::Normal

    # Factor graph properties
    edge_id::String
    node_left::String
    node_right::String
    node_below::String

    function EdgeGaussian(mean::Float64, 
                          precision::Float64, 
                          id::String, 
                          node_l::String, 
                          node_r::String,
                          node_b::String)

        # Check valid precision
        if precision <= 0
            throw("Exception: non-positive precision.")

        # Set recognition distribution parameters
        params["mean"] = mean
        params["precision"] = precision

        # Initial change in entropy
        change_entropy = Inf
        
        # Set graph properties
        edge_id = id
        node_left = node_l
        node_right = node_r
        node_below = node_b

    end
end

function update(edge::Type{EdgeGaussian}, message0, message1)
    "Update recognition distribution as the product of messages"

    # Compute entropy of edge before update
    entropy_old = entropy(edge)

    # Precisions
    precision0 = inv(std(message0)^2)
    precision1 = inv(std(message1)^2)

    # Update variational parameters
    precision = (precision0 + precision1)
    mean = inv(precision)*(precision0*mean(message0) + precision1*mean(message1))

    # Update attributes
    edge.params["precision"] = precision
    edge.params["mean"] = mean

    # Compute new entropy
    entropy_new = entropy(edge)

    # Store change in entropy
    edge.change_entropy = entropy_old - entropy_new

    return Nothing 
end

function entropy(edge::Type{EdgeGaussian})
    "Compute entropy of Gaussian distribution"
    return log(2*pi)/2 + log(edge.params["precision"]^2)/2
end

function message(edge::Type{EdgeGaussian})
    "Outgoing message is updated variational parameters plus change in entropy"
    return (Normal(edge.params["mean"], edge.params["precision"]), edge.change_entropy)
end

function tick()
    "Time progresses one tick and edge should act"

    ..
end