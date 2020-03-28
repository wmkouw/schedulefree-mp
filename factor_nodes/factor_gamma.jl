export FactorGamma


using SpecialFunctions: gamma, digamma
include(joinpath(@__DIR__, "../prob_operations.jl"))
include(joinpath(@__DIR__, "../graph_operations.jl"))
include(joinpath(@__DIR__, "../util.jl"))

mutable struct FactorGamma
    """
    Composite Gamma factor node.

        f(x, α, θ) = Γ(x | α, θ)

    This Gamma distribution is in shape-scale parameterization.

    Examples
    ≡≡≡≡≡≡≡≡≡≡

    # Gamma variable
    julia> γ = VarGamma(:γ);

    # Gamma prior for gamma variable
    julia> f = FactorGamma(:f, out=:γ)

    """

    # Identifiers of edges/nodes in factor graph
    id::Symbol
    variables::Dict{String, Union{Float64, Symbol}}
    time::Union{Integer,Nothing}

    # Reaction parameters
    threshold::Float64
    fired::Bool

    # Additional properties
    verbose::Bool

    function FactorGamma(id::Symbol;
                         out::Union{Float64, Symbol}=0.0,
                         shape::Union{Float64, Symbol}=0.0,
                         scale::Union{Float64, Symbol}=0.0,
                         time::Union{Integer,Nothing}=0,
                         threshold::Float64=0.0,
                         fired::Bool=false,
                         verbose::Bool=false)

        # Keep track of recognition distributions
        variables = Dict{String, Union{Float64, Symbol}}("out" => out,
                                                         "shape" => shape,
                                                         "scale" => scale)

        # Construct instance
        return new(id, variables, time, threshold, fired, verbose)
    end
end

function message(graph::MetaGraph, node::FactorGamma, var_id::Symbol)
    "Compute a message to an edge."

    # Moments of variables
    Ex, Vx = moments(graph, node.variables["out"])
    Eα, Vα = moments(graph, node.variables["shape"])
    Eθ, Vθ = moments(graph, node.variables["scale"])

    # Find which variable belongs to var_id
    var_name = key_from_value(node.variables, var_id)

    if edge_name == "data"

        # Pass message based on belief over parameters
        message = Gamma(Eα, Eθ)

    elseif edge_name == "shape"

        # Pass message based on belief over parameters
        error("Exception: not implemented yet.")

    elseif edge_name == "scale"

        # Pass message based on belief over parameters
        error("Exception: not implemented yet.")

    else
        throw("Exception: variable unknown.")
    end

    return message
end

function act!(graph::MetaGraph, node::FactorGamma, var_id::Symbol)
    "Send out message to connected edge."

    # Compute message
    message_f2v = message(graph, node, var_id)

    # Find edge id
    edge_id = Edge(graph[var_id, :id], graph[node.id, :id])

    # Pass message
    set_prop!(graph, edge_id, :message_factor2var, message_f2v)
end

function react!(graph::MetaGraph, node::FactorGamma)
    "React to incoming messages from edges."

    # Keep track of whether node has fired
    node.fired = false

    # Set of variable ids
    var_ids = Set{Symbol}([x for x in values(node.variables) if isa(x, Symbol)])

    # Loop over variables to check for incoming messages
    for var_id in var_ids

        # Find edge id
        edge = Edge(graph[var_id, :id], graph[node.id, :id])

        # If change in FE is large enough, record variable as updated
        if get_prop(graph, edge, :∇free_energy) >= node.threshold

            # Loop over other variables
            for other_var in setdiff(var_ids, [var_id])

                # Pass message to other variable
                act!(graph, node, other_var)
            end

            # Record that node has fired
            node.fired = true
        end
    end
end
