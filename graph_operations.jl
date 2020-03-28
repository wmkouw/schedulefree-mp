"""
Operations on the factor graph

Wouter Kouw
28-03-2020
"""

import LightGraphs: edges, add_edge!
import MetaGraphs: set_props!


function nodes_t(graph::AbstractMetaGraph, timeslice::Integer; include_notime::Bool=false)
    "List of nodes, by :id, in current timeslice"

    # Preallocate list
    nodes_t = Any[]

    # Filter vertex by :time property
    for node in filter_vertices(graph, :time, timeslice)

        # Push to list
        push!(nodes_t, get_prop(graph, node, :id))
    end

    if include_notime

        # Find nodes without time subscript
        for node in filter_vertices(graph, :time, nothing)

            # Push to list
            push!(nodes_t, get_prop(graph, node, :id))
        end
    end

    return nodes_t
end

function edges(graph::AbstractMetaGraph, node_id::Union{Number, Symbol})
    "Get edges connected to given node"

    # Retrieve node number based on id
    if isa(node_id, Symbol)
        node_id = graph[node_id, :id]
    end

    # Preallocate edges subset
    edges_ = Vector{LightGraphs.SimpleGraphs.AbstractSimpleEdge}()

    # Iterate through edges in graph
    for edge in edges(graph)

        # Check whether edge is connected to target node
        if (edge.src == node_id) | (edge.dst == node_id)
            push!(edges_, edge)
        end
    end
    return edges_
end

function add_edge!(graph::AbstractMetaGraph,
                   edge::Tuple{Int64, Int64};
                   message_factor2var::UnivariateDistribution=Flat(),
                   message_var2factor::UnivariateDistribution=Flat(),
                   ∇free_energy::Float64=Inf)
    "Overload add_edge! to initialize message and belief dictionaries."

    if (edge[1] >= 1) & (edge[2] >= 1)

        # Connect two node
        add_edge!(graph, edge[1], edge[2])

        # Edge id
        edge_id = Edge(edge[1], edge[2])

        # Add message and belief dictionaries to edge
        set_prop!(graph, edge_id, :message_var2factor, message_var2factor)
        set_prop!(graph, edge_id, :message_factor2var, message_factor2var)
        set_prop!(graph, edge_id, :∇free_energy, ∇free_energy)
    end
end

function act!(graph::MetaGraph, node_num::Union{Symbol, Integer})
    "Boiler-plate forwarding"

    # Convert node symbol to node number
    if isa(node_num, Symbol)
        node_num = graph[node_num, :id]
    end

    # Make sure :node is an index
    set_indexing_prop!(graph, :node)

    # Find node based on node number
    node = graph[node_num, :node]

    # Let node act
    act!(graph::MetaGraph, node)

end

function react!(graph::MetaGraph, node_num::Union{Symbol, Integer})
    "Boiler-plate forwarding"

    # Convert node symbol to node number
    if isa(node_num, Symbol)
        node_num = graph[node_num, :id]
    end

    # Make sure :node is an index
    set_indexing_prop!(graph, :node)

    # Find node based on node number
    node = graph[node_num, :node]

    # Let node act
    react!(graph::MetaGraph, node)

end
