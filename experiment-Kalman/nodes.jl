export node_gaussian

struct node_gaussian
"
Gaussian node, used for state transitions or observations.
"
    # Attributes: probability
    var_data
    param_mean
    param_precision

    # Attributes: factor graph architecture
    id
    edge1_id
    edge2_id

    function receive_message_data(message)
        "Store incoming message"

        if istype(message, Delta)

        elseif istype(message, Normal)

        end

        return Nothing
    end

    function receive_message_mean()

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
end
