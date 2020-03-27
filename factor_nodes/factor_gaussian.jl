export FactorGaussian

import Distributions: Normal, Gamma, mean, std, var, params
import DataStructures: Queue, enqueue!, dequeue!
include(joinpath(@__DIR__, "../util.jl"))

mutable struct FactorGaussian
    """
    Composite Gaussian factor node.

        f(y, x, a, u, z) = 𝓝(y | a*x + u, z)

    This is a Gaussian node, but slightly more general: it contains a
    multiplication node with a transition coefficient and an addition node with
    a control coefficient. By default, these are clamped such that a standard
    Gaussian is employed.

    Examples
    ≡≡≡≡≡≡≡≡≡≡

    # States
    julia> x_1 = VarGaussian(:x_1);
    julia> x_2 = VarGaussian(:x_2);

    # Define Gaussian factor for state transition
    julia> f = FactorGaussian(:f, out=:x_2, mean=:x_1)

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

    function FactorGaussian(id::Symbol;
                            out::Union{Float64, Symbol}=0.0,
                            mean::Union{Float64, Symbol}=0.0,
                            transition::Union{Float64, Symbol}=1.0,
                            control::Union{Float64, Symbol}=0.0,
                            precision::Union{Float64, Symbol}=1.0,
                            threshold::Float64=0.0,
                            time::Union{Integer,Nothing}=0,
                            fired::Bool=false,
                            verbose::Bool=false)

        # Keep track of recognition distributions
        variables = Dict{String, Union{Float64, Symbol}}("out" => out,
                                                         "mean" => mean,
                                                         "transition" => transition,
                                                         "control" => control,
                                                         "precision" => precision)

        # Construct instance
        return new(id, variables, time, threshold, fired, verbose)
    end
end

function message(graph::MetaGraph, node::FactorGaussian, var_id::Symbol)
    "Compute a message to an edge."

    # Moments of variables
    Ex, Vx = moments(graph, node.variables["out"])
    Em, Vm = moments(graph, node.variables["mean"])
    Ea, Va = moments(graph, node.variables["transition"])
    Eu, Vu = moments(graph, node.variables["control"])
    Eγ, Vγ = moments(graph, node.variables["precision"])

    # Find which variable belongs to var_id
    var_name = key_from_value(node.variables, var_id)

    if var_name == "out"

        message = Normal(Ea*Em, sqrt(inv(Eγ)))

    elseif var_name == "mean"

        message = Normal(Ex*Ea / (Va + Ea^2), sqrt(inv(Eγ*(Va + Ea^2))))

    elseif var_name == "transition"

        message = Normal(Ex*Em / (Vm + Em^2), sqrt(inv(Eγ*(Vm + Em^2))))

    elseif var_name == "control"

        error("Exception: not implemented yet.")

    elseif var_name == "precision"

        inv_scale = (Vx + Ex^2 - 2*Ex*Ea*Em + (Em^2 + Vm)*(Ea^2 + Va))/2
        message = Gamma(3/2., inv(inv_scale))

    else
        throw("Exception: variable unknown.")
    end

    return message
end

function act!(graph::MetaGraph, node::FactorGaussian, var_id::Symbol)
    "Send out message to connected edge."

    # Compute message
    message_f2v = message(graph, node, var_id)

    # Find edge id
    edge_id = Edge(graph[var_id, :id], graph[node.id, :id])

    # Pass message
    set_prop!(graph, edge_id, :message_factor2var, message_f2v)
end

function react!(graph::MetaGraph, node::FactorGaussian)
    "React to incoming messages from edges."

    # Keep track of whether node fired
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
        else
            println("grad below threshold")
        end
    end
end
