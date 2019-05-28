export GaussianNode

type GaussianNode
"
Gaussian reactive node
"

    # Attributes
    data =
    param_mean =
    param_precision =

    function energy(q::RecognitionDistribution, p::GenerativeDistribution)
        "Compute internal energy of node"

        # Expectations
        Ex = mean(data)
        Em = mean(param_mean)
        Ep = mean(param_precision)

        # -log-likelihood of Gaussian with expected parameters
        return -[-1/2.*log(2*Pi) + log(precision) - 1/2*(Ex - Em)'*EP*(Ex - Em)

    end

    function message_data()
        "Compute outgoing message"

    end

    function message_mean()
        "Compute outgoing message"

    end

    function message_precision()
        "Compute outgoing message"

    end

    function react(DFE)
        "Decide to react based on delta Free Energy"

    end
end
