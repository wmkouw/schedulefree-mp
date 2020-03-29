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
using Plots
pyplot()

# Factor graph components
include(joinpath(@__DIR__, "../factor_nodes/factor_gaussian.jl"))
include(joinpath(@__DIR__, "../factor_nodes/factor_gamma.jl"))
include(joinpath(@__DIR__, "../variables/var_gaussian.jl"))
include(joinpath(@__DIR__, "../variables/var_gamma.jl"))
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
TT = 20

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

	for t = 1:T

		# State transition node
		props_gt = Dict(:id => :g_*t,
					    :time => t,
					    :node => FactorGaussian(:g_*t,
									            out=:x_*t,
		                                        mean=:x_*(t-1),
		                                        precision=inv(process_noise),
		                                        transition=gain))

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
			 							        time=t))

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
F = zeros(T,TT)

# Start message routine
act!(graph, :x_0)

for t = 1:T

	# Report progress
	if mod(t, T/10) == 1
	  println("At iteration "*string(t)*"/"*string(T))
	end

	# Start clock for reactions
	for tt = 1:TT

		# Iterate over all nodes to react
		for node in nodes_t(graph, t)

			# Tell node to react
			react!(graph, node)

			# Check for state variable
			if string(node)[1] == 'x'

				# Retrieve object from node
				node_x = graph[graph[node, :id], :node]

				# Collect free energy
				F[t,tt] = node_x.free_energy
			end
		end
	end
end

CPUtoc()

"""
Visualize experimental results
"""

# Preallocate state vectors
estimated_states = zeros(T,2)

# Loop over time
for t = 1:T

	# Loop over nodes in timeslice
	for node in nodes_t(graph, t)

		# Check for state node
		if string(node)[1] == 'x'

			# Retrieve object from node
			node_x = graph[graph[node, :id], :node]

			# Extract moments of marginals
			estimated_states[t,1] = mean(node_x.marginal)
			estimated_states[t,2] = sqrt(var(node_x.marginal))
		end
	end
end

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
savefig(joinpath(@__DIR__, "viz/exp-fixnumiter_state_estimates.png"))

# Plot free energy
plot(1:T, sum(F, dims=2), color="green", label="")
xlabel!("time (t)")
ylabel!("free energy (F)")
savefig(joinpath(@__DIR__, "viz/exp-fixnumiter_fe-time.png"))

# Plot free energy
plot(1:TT, sum(F, dims=1)', color="green", label="")
xlabel!("#iterations")
ylabel!("free energy (F)")
savefig(joinpath(@__DIR__, "viz/exp-fixnumiter_fe-iters.png"))
