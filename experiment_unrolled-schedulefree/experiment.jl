# Reactive message-passing in linear Gaussian dynamical system
# Experiment to pass messages without a schedule
#
# Wouter Kouw, BIASlab
# 12-08-2019

using Revise
using Distributions
using DataStructures
using LightGraphs
using MetaGraphs
using Plots
gr()

# Factor graph components
include("../nodes/node_gaussian.jl")
include("../nodes/transition_gaussian.jl")
include("../nodes/likelihood_gaussian.jl")
include("../edges/edge_gaussian.jl")
include("../edges/edge_delta.jl")
include("../util.jl")

# Data
include("../gen_data.jl")

"""
Experiment parameters
"""

# Signal time horizon
T = 3

# Reaction-time clock
TT = 20

# Reaction heuristics
heuristics = Dict{String,Any}("backwards_in_time" => false)

# Known transition and observation matrices
gain = 0.8
emission = 1.0

# Known noises (variance form)
process_noise = 2.0
measurement_noise = 0.5

# Parameters for state x_0
mean_0 = 0.0
precision_0 = 1.0

# Generate data
observed, hidden = gendata_LGDS(gain,
                                emission,
                                process_noise,
                                measurement_noise,
                                mean_0,
                                precision_0,
                                time_horizon=T)

"""
Model/graph specification

This assumes the following type of models
p(y_{1:T}, x_{0:T} | u_{1:T}) = p(x_0) Π_t p(y_t, x_t | x_{t-1})

In other words, Markov chains of time-slices of a state-space models.
Below, we specify the following model through the time-slice subgraph

x_t-1    _       x_t
... --∘---->|_|----->|=|-----> ...
        g_t       |
                  |
                 |_| f_t
                  |
                  ∘
                 y_t

x_t-1 = previous state edge
g_t = state transition edge
x_t = current state edge
f_t = likelihood node
y_t = observation node
"""

# Start graph
global graph = MetaGraph(SimpleGraph(13))

# Prior state
x_0 = EdgeGaussian("x_0", time=0, mean=mean_0, precision=precision_0)
set_props!(graph, 1, Dict{Symbol,Any}(:object => :x_0, :id => "x_0"))

# State transition
g_1 = TransitionGaussian("g_1", time=1, edge_mean="x_0", edge_data="x_1", edge_precision=inv(process_noise), heuristics=heuristics)
set_props!(graph, 2, Dict{Symbol,Any}(:object => :g_1, :id => "g_1"))
add_edge!(graph, 1, 2) # x_t-1 -- g_t

# State
x_1 = EdgeGaussian("x_1", time=1, mean=0.0, precision=1.0)
set_props!(graph, 3, Dict{Symbol,Any}(:object => :x_1, :id => "x_1"))
add_edge!(graph, 2, 3) # g_t -- x_t

# Observation likelihood
f_1 = LikelihoodGaussian("f_1", time=1, edge_mean="x_1", edge_data="y_1", edge_precision=inv(measurement_noise), heuristics=heuristics)
set_props!(graph, 4, Dict{Symbol,Any}(:object => :f_1, :id => "f_1"))
add_edge!(graph, 3, 4) # x_t -- f_t

# Observation
y_1 = EdgeDelta("y_1", time=1, observation=observed[1])
set_props!(graph, 5, Dict{Symbol,Any}(:object => :y_1, :id => "y_1"))
add_edge!(graph, 4, 5) # f_t -- y_t

# State transition
g_2 = TransitionGaussian("g_2", time=2, edge_mean="x_1", edge_data="x_2", edge_precision=inv(process_noise), heuristics=heuristics)
set_props!(graph, 6, Dict{Symbol,Any}(:object => :g_2, :id => "g_2"))
add_edge!(graph, 3, 6) # x_t-1 -- g_t

# State
x_2 = EdgeGaussian("x_2", time=2, mean=0.0, precision=1.0)
set_props!(graph, 7, Dict{Symbol,Any}(:object => :x_2, :id => "x_2"))
add_edge!(graph, 6, 7) # g_t -- x_t

# Observation likelihood node
f_2 = LikelihoodGaussian("f_2", time=2, edge_mean="x_2", edge_data="y_2", edge_precision=inv(measurement_noise), heuristics=heuristics)
set_props!(graph, 8, Dict{Symbol,Any}(:object => :f_2, :id => "f_2"))
add_edge!(graph, 7, 8) # x_t -- f_t

# Observation
y_2 = EdgeDelta("y_2", time=2, observation=observed[2])
set_props!(graph, 9, Dict{Symbol,Any}(:object => :y_2, :id => "y_2"))
add_edge!(graph, 8, 9) # f_t -- y_t

# State transition
g_3 = TransitionGaussian("g_3", time=3, edge_mean="x_2", edge_data="x_3", edge_precision=inv(process_noise), heuristics=heuristics)
set_props!(graph, 10, Dict{Symbol,Any}(:object => :g_3, :id => "g_3"))
add_edge!(graph, 7, 10) # x_t-1 -- g_t

# State
x_3 = EdgeGaussian("x_3", time=3, mean=0.0, precision=1.0)
set_props!(graph, 11, Dict{Symbol,Any}(:object => :x_3, :id => "x_3"))
add_edge!(graph, 10, 11) # g_t -- x_t

# Observation likelihood node
f_3 = LikelihoodGaussian("f_3", time=3, edge_mean="x_3", edge_data="y_3", edge_precision=inv(measurement_noise), heuristics=heuristics)
set_props!(graph, 12, Dict{Symbol,Any}(:object => :f_3, :id => "f_3"))
add_edge!(graph, 11, 12) # x_t -- f_t

# Observation
y_3 = EdgeDelta("y_3", time=3, observation=observed[3])
set_props!(graph, 13, Dict{Symbol,Any}(:object => :y_3, :id => "y_3"))
add_edge!(graph, 12, 13) # f_t -- y_t

# Ensure vertices can be recalled from given id
set_indexing_prop!(graph, :id)
set_indexing_prop!(graph, :object)

"""
Start reactive message-passing procedure
"""

# Preallocation
estimated_states = zeros(T, TT, 2)

# Start message routine
act(x_0, belief(x_0), Inf, graph)

# Start clock for reactions
for tt = 1:TT

      # Report progress
      if mod(tt, TT/10) == 1
          println("Reaction clock "*string(tt)*"/"*string(TT))
      end

      # Each node reacts
      react(g_1, graph)
      react(x_1, graph)
      react(f_1, graph)
      react(y_1, graph)
      react(g_2, graph)
      react(x_2, graph)
      react(f_2, graph)
      react(y_2, graph)
      react(g_3, graph)
      react(x_3, graph)
      react(f_3, graph)
      react(y_3, graph)

      # Write out estimated state parameters
      estimated_states[1, tt, 1] = x_1.mean
      estimated_states[1, tt, 2] = sqrt(1/x_1.precision)
      estimated_states[2, tt, 1] = x_2.mean
      estimated_states[2, tt, 2] = sqrt(1/x_2.precision)
      estimated_states[3, tt, 1] = x_3.mean
      estimated_states[3, tt, 2] = sqrt(1/x_3.precision)
end

"""
Visualize experimental results
"""

# Visualize final estimates over time
plot(hidden[2:end], color="red", label="states")
plot!(estimated_states[:,end,1], color="blue", label="estimates")
plot!(estimated_states[:,end,1],
      ribbon=[estimated_states[:,end,2], estimated_states[:,end,2]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
scatter!(observed, color="black", label="observations")
savefig(pwd()*"/experiment_unrolled-schedulefree/viz/state_estimates.png")

# Visualize state estimate trajectories
plot(estimated_states[1,:,1], color="blue", label="estimates")
plot!(estimated_states[1,:,1],
      ribbon=[estimated_states[1,:,2], estimated_states[1,:,2]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
savefig(pwd()*"/experiment_unrolled-schedulefree/viz/trajectory_x1.png")

# Visualize state estimate trajectories
for t = 1:T
      plot(estimated_states[t,:,1], color="blue", label="estimates")
      plot!(estimated_states[t,:,1],
            ribbon=[estimated_states[t,:,2], estimated_states[t,:,2]],
            linewidth=2,
            color="blue",
            fillalpha=0.2,
            fillcolor="blue",
            label="")
      savefig(pwd()*"/experiment_unrolled-schedulefree/viz/trajectory_x"*string(t)*".png")
end
