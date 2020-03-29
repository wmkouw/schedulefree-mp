# Reactive message-passing in a linear Gaussian dynamical system
# Experiment to infer states
# Termination based on local Free Energy
#
# Wouter Kouw, BIASlab
# 26-03-2020

using Random
using Distributions
using LightGraphs
using MetaGraphs
using CPUTime

# Factor graph components
include(joinpath(@__DIR__, "../factor_nodes/factor_gaussian.jl"))
include(joinpath(@__DIR__, "../variables/var_gaussian.jl"))
include(joinpath(@__DIR__, "../variables/var_delta.jl"))
include(joinpath(@__DIR__, "../prob_operations.jl"))
include(joinpath(@__DIR__, "../graph_operations.jl"))
include(joinpath(@__DIR__, "../util.jl"))

# Data
include(joinpath(@__DIR__, "../gen_data.jl"))

"""
Experiment parameters
"""

# Signal time horizon
T = 50

# Reaction-time clock
TT = 10

# Free Energy threshold
fe_threshold = 1e-3

# Known transition and observation matrices
gain = 0.8
emission = 1.0

# Known noises (variance form)
process_noise = 2.0
measurement_noise = 0.5

# Parameters for state x_0
px0 = Normal(0.0, 1.0)

# Generate data
Random.seed!(256)
observed, hidden = gendata_LGDS(gain,
                                emission,
                                process_noise,
                                measurement_noise,
                                mean(px0),
                                inv(var(px0)),
                                time_horizon=T)

"""
Model/graph specification

This assumes the following type of models
p(y_{1:T}, x_{0:T}) = p(x_0) Π_t p(y_t | x_t) p(x_t | x_{t-1})

In other words, Markov chains of time-slices of a state-space models.
Below, we specify the following model through the time-slice subgraph
     _____      ___      ___
... (x_t-1)--->[g_t]--->(x_t)  ...
                          |
                         _|_
                        [f_t]
                          |
                         _|_
                       [[y_t]]

x_t-1 = previous state edge
g_t = state transition node
x_t = current state edge
f_t = likelihood node
y_t = observation node

---------------------------------
#TODO: Graph specification using a graphical user interface
"""

CPUtic()

# Start graph
global graph = MetaGraph(SimpleGraph(4*T+1))

let

	# Set initial state prior
	set_props!(graph, 1, Dict(:id => :x_0,
							  :time => 0,
							  :node => VarGaussian(:x_0, marginal=px0, time=0)))

    # Connect initial state prior to first state transition
    add_edge!(graph, (1, 2))

	# Start node numbering
	node_num = 1

	# Build whole graph (ie all t)
	for t = 1:T

		# State transition node
		props_gt = Dict(:id => :g_*t,
					    :time => t,
						:threshold => fe_threshold,
					    :node => FactorGaussian(:g_*t,
									            out=:x_*t,
		                                        mean=:x_*(t-1),
		                                        precision=inv(process_noise),
		                                        transition=gain,
												threshold=fe_threshold))

		# Current state
		props_xt = Dict(:id => :x_*t,
					    :time => t,
					    :node => VarGaussian(:x_*t, time=t))

		# Observation likelihood node
		props_ft = Dict(:id => :f_*t,
					    :time => t,
					    :node => FactorGaussian(:f_*t,
			 						            out=:y_*t,
			 							        mean=:x_*t,
			 							        precision=inv(measurement_noise),
			 							        transition=emission,
			 							        time=t,
												threshold=fe_threshold))

		# Observation
		props_yt = Dict(:id => :y_*t,
					    :time => t,
					    :node => VarDelta(:y_*t, observed[t], time=t))

		# Add properties to nodes in current time-slice
		set_props!(graph, node_num+1, props_gt)
		set_props!(graph, node_num+2, props_xt)
		set_props!(graph, node_num+3, props_ft)
		set_props!(graph, node_num+4, props_yt)

		# Add edges between factors and variables
		add_edge!(graph, (node_num-2, node_num+1)) # x_t-1 -- g_t
		add_edge!(graph, (node_num+1, node_num+2)) # g_t -- x_t
		add_edge!(graph, (node_num+2, node_num+3)) # x_t -- f_t
		add_edge!(graph, (node_num+3, node_num+4)) # f_t -- y_t

		# Increment node number
		node_num += 4
	end
end

# Ensure vertices can be recalled from given id
set_indexing_prop!(graph, :id)

"""
Inference: filtering
"""

# Preallocate energy
LFE = Vector{Array{Float64,1}}(undef,T)

# Start message routine
act!(graph, :x_0)

for t = 1:T

	# Report progress
	# if mod(t, T/5) == 1
  	println("At iteration "*string(t)*"/"*string(T))
	# end

	# Fire state transiton from previous time-point
	act!(graph, graph[graph[:g_*t, :id], :node], :x_*t)

	# Start clock for reactions
	tt = 0
	LFE_t = Float64[]
	nodes_fired = true
	while nodes_fired | (tt <= 3)

		# Re-start node fired check
		nodes_fired = false

		# Count iterations
		tt += 1

		# Iterate over all nodes to react
		for node_id in nodes_t(graph, t)

			# Retrieve object from node
			node = graph[graph[node_id, :id], :node]

			# Tell node to react
			react!(graph, node_id)

			# Check for state variable
			if typeof(node) == VarGaussian

				# Collect free energy
				push!(LFE_t, node.free_energy)
			end

			# Check for factor node
			if typeof(node) == FactorGaussian

				# Record whether node fired
				nodes_fired |= node.fired
			end
		end
	end

	# Record LFE
	LFE[t] = LFE_t
end

CPUtoc()

"""
Get estimates from graph
"""

# Preallocate state vectors
estimated_states = zeros(T,2)
F = zeros(T,)
A = zeros(T,)

# Loop over time
for t = 1:T

	# Loop over nodes in timeslice
	for node_id in nodes_t(graph, t)

		# Retrieve object from node
		node = graph[graph[node_id, :id], :node]

		# Check for state node
		if typeof(node) == VarGaussian

			# Extract moments of marginals
			estimated_states[t,1] = mean(node.marginal)
			estimated_states[t,2] = sqrt(var(node.marginal))

			# Track final FE
			F[t] = node.free_energy
		end

		# Check for state node
		if typeof(node) == VarDelta

			# Track final accuracy (logpdf of observation under message)
			A[t] = node.pred_error
		end
	end
end

"""
Visualize experimental results
"""

using Plots
pyplot()

# Plot estimated states
scatter(1:T, observed, color="black", label="observations")
plot!(hidden[2:end], color="red", label="true states")
plot!(estimated_states[:,1], color="blue", label="inferred")
plot!(estimated_states[:,1],
	  ribbon=[estimated_states[:,2], estimated_states[:,2]],
	  linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
	  label="")
xlabel!("time (t)")
savefig(joinpath(@__DIR__, "viz/exp-lfeterm_state_estimates.png"))

# Plot free energy
plot(1:T, F, color="green", label="")
xlabel!("time (t)")
ylabel!("free energy (F)")
savefig(joinpath(@__DIR__, "viz/exp-lfeterm_fe-time.png"))

# Plot accuracy
plot(1:T, A, color="green", label="")
xlabel!("time (t)")
ylabel!("wpred err (-log μ(x))")
savefig(joinpath(@__DIR__, "viz/exp-lfeterm_acc-time.png"))
