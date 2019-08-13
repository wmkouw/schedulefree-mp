export EdgeDelta

using Distributions: Normal, Uniform, logpdf

mutable struct EdgeDelta
    """
    Edge for an observation
    """
    # Distribution parameters
    observation::Float64
    free_energy::Float64

    # Message bookkeeping
    messages::Dict{String, Normal}

    # Factor graph properties
    id::String

    function EdgeDelta(id; observation=Uniform(-Inf,Inf))

        # Initialize messages
        messages = Dict{String, Any}()

        # Initialize free energy
        free_energy = Inf

        self = new(id, observation, free_energy, messages)
        return self
    end
end

function entropy(edge::EdgeDelta)
    "An observed variable has no entropy"
    return 0.0
end

function free_energy(edge::EdgeGaussian, graph::MetaGraph)
    """
    Free energy at observation node is node energy evaluated at observation,
    in other words, the precision-weighted prediction error.
    """

    # Extract id of likelihood node
    N = neighbors(graph, graph[edge.id, :id])
    node_id = [graph[n, :id] for n in N]

    # Collect node variable via graph
    node = get_prop!(graph, graph[node_id, :id], :object)

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

function act(edge::EdgeDelta, message, old_free_energy, graph::MetaGraph)
    "Pass observation to likelihood node."

    # Get edge name from edge id
    edge_name = key_from_value(edge.id)

    # Extract likelihood node id
    n = neighbors(graph, graph[edge.id, :id])
    node = get_prop!(graph, graph[graph[n, :id], :id], :object)

    # Update belief at node
    node.beliefs[edge_name] = belief

    # Compute change in free energy due to passing message
    delta_free_energy = free_energy(edge) - free_energy

    # Tell node that it has received a new message
    enqueue!(node.incoming, (edge.id, delta_free_energy))

    return Nothing
end

function react(edge::EdgeDelta, graph::MetaGraph)
    "React to incoming messages"

    # Estimate free energy
    old_free_energy = free_energy(edge)

    # Message from edge to nodes
    act(edge, message(edge), old_free_energy, graph)

    return Nothing
end
