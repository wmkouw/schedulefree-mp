export EdgeDelta

using Distributions: Normal, Uniform, logpdf

mutable struct EdgeDelta
    """
    Edge for an observation
    """

    # Factor graph properties
    id::String
    block::Bool

    # Distribution parameters
    observation::Float64
    free_energy::Float64

    # Message bookkeeping
    messages::Dict{String, Any}

    function EdgeDelta(id; block=false, observation=NaN)

        # Initialize messages
        messages = Dict{String, Any}()

        # Initialize free energy
        free_energy = Inf

        self = new(id, block, observation, free_energy, messages)
        return self
    end
end

function entropy(edge::EdgeDelta)
    "An observed variable has no entropy."
    return 0.0
end

function gradient_entropy(edge:EdgeDelta)
    "Gradient of entropy for observed variable is 0."
    return 0.0
end

function free_energy(edge::EdgeDelta, graph::MetaGraph)
    """
    Free energy at observation node is node energy evaluated at observation,
    in other words, the precision-weighted prediction error.
    """

    # Extract id of likelihood node
    node_id = neighbors(graph, graph[edge.id, :id])[1]

    # Collect node variable via graph
    node = eval(get_prop(graph, node_id, :object))

    # Add to total energy
    return energy(node) - entropy(edge)

end

function prediction_error(edge::EdgeDelta, message)
    "Compute precision-weighted prediction error"
    return -logpdf(message, edge.observation)
end

function belief(edge::EdgeDelta)
    "Edge is distribution"
    return edge.observation
end

function act(edge::EdgeDelta, belief, old_free_energy, graph::MetaGraph)
    "Pass observation to likelihood node."

    # Extract connecting node
    node_id = neighbors(graph, graph[edge.id, :id])[1]

    # Call node from id
    node = eval(get_prop(graph, graph[graph[node_id, :id], :id], :object))

    # Get edge name from edge id
    edge_name = key_from_value(node.connected_edges, edge.id)

    # Update belief at node
    node.beliefs[edge_name] = belief

    # Compute change in free energy due to passing message
    delta_free_energy = free_energy(edge, graph) - old_free_energy

    # Tell node that it has received a new message
    enqueue!(node.incoming, (edge.id, delta_free_energy))

    return Nothing
end

function react(edge::EdgeDelta, graph::MetaGraph)
    "React to incoming messages"

    # Estimate free energy
    old_free_energy = free_energy(edge, graph)

    # Message from edge to nodes
    act(edge, belief(edge), old_free_energy, graph)

    return Nothing
end
