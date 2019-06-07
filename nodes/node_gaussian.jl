export NodeGaussian

using Distributions: Normal
using DataStructures: Queue, enqueue!, dequeue!

mutable struct NodeGaussian

    # Attributes: factor graph architecture
    id::String
    edges::Dict{String, Symbol}

    # List of messages
    messages::Dict{String, Queue{Normal}}

    # Attributes: matrices for Kalman filter
    A::Float64
    Q::Float64

    function NodeGaussian(edge_data_id::Symbol,
                          edge_mean_id::Symbol,
                          transition::Float64,
                          precision::Float64,
                          id::String)

        # Connect node to specific edges
        edges = Dict{String, Symbol}("data" => edge_data_id, "mean" => edge_mean_id)

        # Keep track of incoming messages
        messages = Dict{String, Queue{Normal}}("data" => Queue{Normal}(), "mean" => Queue{Normal}())
        
        # Create instance
        self = new(id, edges, messages)

        # Transition matrix
        self.A = transition;
        self.Q = precision;

        return self
    end
end


function energy(node::Type{NodeGaussian})
    "Compute internal energy of node"

    # Expected mean
    Em = node.A*node.edges["mean"].params[1]

    # Expected data
    Ex = node.edges["data"].params[1]

    # -log-likelihood of Gaussian with expected parameters
    return -[- 1/2 *log(2*pi) + log(node.Q) - 1/2 *(Ex - Em)'*node.Q*(Ex - Em)]

end

function react()
    "Decide to react based on delta Free Energy"

    # Check for incoming messages
    
end
