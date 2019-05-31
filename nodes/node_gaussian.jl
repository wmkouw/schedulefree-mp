export NodeGaussian

"""
Description:

    A Gaussian with mean-precision parameterization:

    f(out,m,w) = 𝒩(out|m,w) = (2π)^{-D/2} |w|^{1/2} exp(-1/2 (out - m)' w (out - m))

Interfaces:

    1. out
    2. m (mean)
    3. w (precision)

Construction:

    GaussianMeanPrecision(out, m, w, id=:some_id)
"""
mutable struct NodeGaussian

    # Attributes: probability
    var_data
    param_mean
    param_precision

    # Attributes: factor graph architecture
    id
    edge1_id
    edge2_id

    function NodeGaussian()
        return self
    end
end

function receive_message_data(message)
    "Store incoming message"

    if istype(message, Delta)

    elseif istype(message, Normal)

    end

    return Nothing
end

function receive_message_mean()
    
end

function energy(q::RecognitionDistribution, p::GenerativeDistribution)
    "Compute internal energy of node"

    # Expectations
    Em = mean(param_mean)
    Ep = mean(param_precision)

    # -log-likelihood of Gaussian with expected parameters
    return -[-1/2.*log(2*Pi) + log(precision) - 1/2*(Ex - Em)'*EP*(Ex - Em)

end

function react(DFE)
    "Decide to react based on delta Free Energy"

end
