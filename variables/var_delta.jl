export VarDelta


mutable struct VarDelta
    """
    Variable clamped to an observation
    """

    # Node properties
    id::Symbol
    marginal::Delta
    time::Union{Integer,Nothing}
    block::Bool

    # Keep track of prediction error
    pred_error::Float64
	Δpred_error::Float64

    function VarDelta(id::Symbol,
					  observed_value::Float64;
					  time::Union{Integer,Nothing}=0,
					  block=false)

        # Cast observed value to Delta distrbution
        marginal = Delta(observed_value)

		# Initialize prediction error
		pred_error = Inf
		Δpred_error = Inf

        return new(id, marginal, time, block, pred_error, Δpred_error)
    end
end

function entropy(variable::VarDelta)
    "An observed variable has no entropy."
    return 0.0
end

function gradient_entropy(variable::VarDelta)
    "Gradient of entropy for observed variable is 0."
    return 0.0
end

function prediction_error!(graph::MetaGraph, variable::VarDelta)
    "Compute prediction error of observation"

	# Reset prediction error
	pred_error_old = variable.pred_error

	# Find edge connected to node
	edge = edges(graph, graph[variable.id, :id])[1]

	# Extract message from factor to variable
	message_f2v = get_prop(graph, edge, :message_factor2var)

	# Compute and add prediction error
	variable.pred_error = -logpdf(message_f2v, mean(variable.marginal))

	# Compute change in prediction error
	variable.Δpred_error = norm(pred_error_old - variable.pred_error)
end

function belief(variable::VarDelta)
    "Edge is distribution"
    return Delta(variable.observation)
end

function act!(graph::MetaGraph, variable::VarDelta)
    "Pass message to edge with likelihood node"

    # Check each connected edge
    for edge in edges(graph, graph[variable.id, :id])

        # Check for incoming
		set_prop!(graph, edge, :message_var2factor, variable.marginal)
		set_prop!(graph, edge, :Δfree_energy, variable.Δpred_error)
    end
end

function react!(graph::MetaGraph, variable::VarDelta)
    "React to incoming messages"

    # Compute local free energy
    prediction_error!(graph, variable)

    # Message from edge to nodes
    act!(graph, variable)
end
