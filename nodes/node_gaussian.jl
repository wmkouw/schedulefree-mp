export NodeGaussian

using Distributions

mutable struct NodeGaussian

    # Attributes: factor graph architecture
    node_id::String
    edge_data_id::String
    edge_mean_id::String

    # Attributes: incoming messages
    message_data::Type{Normal}
    message_mean::Type{Normal}

    # Attributes: matrices for Kalman filter
    A::Float64
    Q::Float64

    function NodeGaussian(edge_data_id::String,
                          edge_mean_id::String,
                          transition::Float64,
                          precision::Float64,
                          id::String)
        
        # Set graph properties
        node_id = id
        edge_data_id = edge_data_id
        edge_mean_id = edge_mean_id

        # Transition matrix
        A = transition
        Q = precision

        # Initialize messages
        message_data = Normal(0, 1)
        message_mean = Normal(0, 1)
    end
end


function energy(node::Type{NodeGaussian})
    "Compute internal energy of node"

    # Expected mean
    Em = node.A*mean(node.message_mean)

    # Expected data
    Ex = mean(node.message_data)

    # -log-likelihood of Gaussian with expected parameters
    return -[- 1/2 *log(2*pi) + log(node.Q) - 1/2 *(Ex - Em)'*node.Q*(Ex - Em)]

end

function react()
    "Decide to react based on delta Free Energy"
end
