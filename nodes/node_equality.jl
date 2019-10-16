export NodeEquality

using DataStructures: Queue, enqueue!, dequeue!
include("../util.jl")

mutable struct NodeEquality
    """
    Equality node.

    This node has the function:
    f(x,y,z) = δ(z - x) δ(z - y)
           ___
    x --->[_f_]---> z
            ^
            |
            y
    """

    # Identifiers of edges/nodes in factor graph
    id::String
    time::Int64
    connected_edges::Dict{String, String}

    # Reaction parameters
    incoming::Queue{Tuple}
    beliefs::Dict{String, Any}
    energies::Dict{String, Float64}

    function NodeEquality(id; time=0, edges=[])

        # Keep track of recognition distributions
        connected_edges = Dict{String, String}()
        for edge in edges
            # Edge name is also edge key
            connected_edges[edge] = edge
        end

        # Initialize beliefs for all edges
        beliefs = Dict{String, Any}()

        # Initialize queue for incoming messages
        incoming = Queue{Tuple}()

        # Initialize energy placeholder
        energies = Dict{String,Float64}()
        for edge in edges
            energies[edge] = Inf
        end

        # Create instance
        self = new(id, time, connected_edges, incoming, beliefs, energies)
        return self
    end
end

function energy(node::NodeEquality)
    "Pass on energy from incoming messages."
    # Return the sum of stored energies
    return sum([node.energies[key] for key in keys(node.energies)])
end

function grad_energy(node::NodeEquality, edge_id::String)
    "Pass on energy gradients from other connected nodes."
    #TODO (probably via Automatic Differentiation)
    # Find set of other edges
    other_edges = setdiff(keys(node.energies), [edge_id])

    return 1.0, 1.0
end

function message(node::NodeEquality, edge_id::String)
    """
    Pass on messages from other edges.

    The message to z is computed by ∫ qx(x) qy(y) log[f(x,y,z)] dx dy. The
    sifting property of the delta distributions turns this integral into:
        qx(z)⋅qy(z).
    """

    # Find set of other edges
    other_edges = setdiff(keys(node.beliefs), [edge_id])

    # Construct array of incoming messages from other edges
    incoming_messages = [node.beliefs[edge] for edge in other_edges]

    # Outgoing message is the product of the incoming messages
    outgoing_message = prod(incoming_messages)

    # Pass message
    return outgoing_message
end

function act(node::NodeEquality, edge_id::String, graph::MetaGraph)
    "Send out message for one of the connecting edges"

    # Compute message for a particular edge
    outgoing_message = message(node, edge_id)

    # Extract edge from graph
    edge = eval(Symbol(edge_id))

    # Check if edge is blocked
    if !edge.block

        # Pass message to edge
        edge.messages[node.id] = outgoing_message
    end

    return Nothing
end

function react(node::NodeEquality, graph::MetaGraph)
    "React to incoming messages from edges"

    # Find edges attached to node
    N = neighbors(graph, graph[node.id, :id])
    edge_ids = [graph[n, :id] for n in N]

    # Loop over all edges that have produced incoming messages
    for n = 1:length(node.incoming)

        # Extract the id of the edge and the communicated delta free energy
        edge_id, delta_free_energy = dequeue!(node.incoming)

        # Store incoming energy
        node.energies[edge_id] = delta_free_energy

        # Act on other edges than incoming one
        for edge_out in setdiff(Set(edge_ids), Set([edge_id]))
            act(node, edge_out, graph)
        end
    end

    return Nothing
end
