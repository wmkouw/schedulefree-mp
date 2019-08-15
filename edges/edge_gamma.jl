export EdgeGamma

using Distributions: Gamma, mean, params
using DataStructures: Queue, enqueue!, dequeue!
using SpecialFunctions: gamma, digamma

mutable struct EdgeGamma
    "
    Edge with a Gamma recognition distribution
    "
    # Recognition distribution parameters
    shape::Float64
    rate::Float64
    free_energy::Float64

    # Message bookkeeping
    messages::Dict{String, Gamma}

    # Factor graph properties
    id::String

    function EdgeGamma(id; shape=1.0, rate=1.0, free_energy=1e12)

        # Check valid parameters
        if shape < 0
            throw("Exception: shape parameter should be larger than 0.")
        end
        if rate < 0
            throw("Exception: rate parameter should be larger than 0.")
        end

        # Initialize messages
        messages = Dict{String, Gamma}()

        # Construct instance
        self = new(shape, rate, free_energy, messages, id)
        return self
    end
end

function update(edge::EdgeGamma)
    "Update recognition distribution as the product of messages"

    new_shape = 0
    new_rate = 0

    for key in keys(edge.messages)

        # Extract parameters
        shape, rate = params(edge.messages[key])

        new_shape += shape
        new_rate += rate
    end

    # Correct shape update
    edge.shape = new_shape - 1
    edge.rate = new_rate

    return Nothing
end

function belief(edge::EdgeGamma)
    "Outgoing message"
    return Gamma(edge.shape, edge.rate)
end

function entropy(edge::EdgeGamma)
    "Compute entropy of Gamma distribution"

    # Parameters
    a = edge.shape
    b = edge.rate

    # Entropy of a univariate Gamma
    return a - log(b) + log(gamma(a)) + (1-a)*digamma(a)
end

function free_energy(edge::EdgeGamma, graph::MetaGraph)
    "Compute free energy of edge and connecting nodes"

    # Extract connecting nodes
    N = neighbors(graph, graph[edge.id, :id])
    node_ids = [graph[n, :id] for n in N]

    # Initialize node energies
    U = 0

    # Update marginal at connected nodes
    for node_id in node_ids

        # Collect node variable via graph
        node = eval(get_prop(graph, graph[node_id, :id], :object))

        # Add to total energy
        U += energy(node)
    end

    # Compute own entropy
    H = entropy(edge)

    # Return free energy
    return U - H
end

function act(edge::EdgeGamma,
             belief::Gamma,
             delta_free_energy::Float64,
             graph::MetaGraph)
    "Pass belief to connected nodes."

    # Extract connecting nodes
    N = neighbors(graph, graph[edge.id, :id])
    node_ids = [graph[n, :id] for n in N]

    # Update marginal at connected nodes
    for node_id in node_ids

        # Collect node variable via graph
        node = eval(get_prop(graph, graph[node_id, :id], :object))

        # Get edge name from edge id
        edge_name = key_from_value(node.connected_edges, edge.id)

        # Update belief distribution
        node.beliefs[edge_name] = belief

        # Tell node that it has received a new message
        enqueue!(node.incoming, (edge.id, delta_free_energy))
    end

    return Nothing
end

function react(edge::EdgeGamma, graph::MetaGraph)
    "React to incoming messages"

    # Update variational distribution
    update(edge)

    # Compute free energy after update
    new_free_energy = free_energy(edge, graph)

    # Compute change in free energy after update
    delta_free_energy = new_free_energy - edge.free_energy

    # Message from edge to nodes
    act(edge, belief(edge), delta_free_energy, graph)

    # Store new free energy
    edge.free_energy = new_free_energy

    return Nothing
end
