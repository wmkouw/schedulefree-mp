export edge_gaussian

using LinearAlgebra: inv
using Distributions: Normal, mean, std

struct Delta
    "
    Dirac delta distribution
    "
    data

    function probability(x)
        "Non-zero for observed value"
        if x == data
            return 1
        else
            return 0
        end
    end
end

struct edge_gaussian
    "
    Edge with a Gaussian recognition distribution
    "
    # Attributes: parameters recognition, incoming messages
    params = Dict{String, Float64}([("mean", 0), ("precision", 1)])
    params_old = Dict{String, Float64}([("mean", 0), ("precision", 1)])
    message_l = Normal(μ=0, σ=1)
    message_r = Normal(μ=0, σ=1)

    function receive_left(message)
        "Update message attribute"
        message_l = Normal(mean(message), std(message))
        return Nothing
    end

    function receive_right(message)
        "Update message attribute"
        message_r = Normal(mean(message), std(message))
        return Nothing
    end

    function update(message_left, message_right)
        "Update recognition distribution as the product of messages"

        # Store current parameters as old
        params_old["mean"] = params["mean"]
        params_old["precision"] = params["precision"]

        # Precisions
        precision_l = inv(std(message_l)^2)
        precision_r = inv(std(message_r)^2)

        # Update variational parameters
        precision = (precision_l + precision_r)
        mean = inv(precision)*(precision_l*mean(message_l) + precision_r*mean(message_r))

        # Update attributes
        params["precision"] = precision
        params["mean"] = mean

        return Nothing
    end

    function entropy(params)
        "Compute entropy of Gaussian distribution"
        return log(2*pi)/2 + log(params["precision"]^2)/2
    end

    function D_entropy(params_new, params_old)
        "Change in entropy for update to variational params"
        return entropy(params_new) - entropy(params_old)
    end

    function message()
        "Outgoing message is updated variational parameters plus change in entropy"

        # Change in entropy
        DH = D_entropy(params, params_old)

        return Normal(params["mean"], params["precision"]), DH
    end
end

struct edge_delta
    "
    Edge for an observation
    "
    # Attribute
    data

    function message()
        "Outgoing message is updated variational parameters"
        return Delta(data)
    end
end