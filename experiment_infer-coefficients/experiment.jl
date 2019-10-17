# Reactive message-passing in linear Gaussian dynamical system.
# Experiment to infer transition coefficients.
#
# Wouter Kouw, BIASlab
# 15-08-2019

using Revise
using Distributions
using DataStructures
using LightGraphs, MetaGraphs
using Plots
pyplot()

# Factor graph components
include("../nodes/node_gamma.jl")
include("../nodes/node_gaussian.jl")
include("../nodes/node_equality.jl")
include("../nodes/transition_gaussian.jl")
include("../nodes/likelihood_gaussian.jl")
include("../edges/edge_gaussian.jl")
include("../edges/edge_gamma.jl")
include("../edges/edge_delta.jl")
include("../util.jl")

# Data
include("../gen_data.jl")

"""
Experiment parameters
"""

# Signal time horizon
T = 500

# Reaction-time clock
TT = 10

# Known transition and observation matrices
gain = 0.8
emission = 1.0

# Noise parameters (variance form)
measurement_noise = 0.3
process_noise = 1.4

# Clamped parameters (mean-precision, shape-scale form)
x0_params = [0.0, 0.1]
γ0_params = [1.0, 1.0]
a0_params = [0.5, 1.0]

# Generate data
observed, hidden = gendata_LGDS(gain,
                                emission,
                                process_noise,
                                measurement_noise,
                                x0_params[1],
                                x0_params[2],
                                time_horizon=T)

"""
Model/graph specification

This assumes the following type of models
p(y_{1:T}, x_{0:T} | u_{1:T}) = p(x_0) Π_t p(y_t, x_t | x_{t-1})

In other words, Markov chains of time-slices of a state-space models.
Below, we specify the following model through the time-slice subgraph

               e_a
...   (a')---->[=]-----------(a'')     ...
                |
                |    e_γ
...   (γ')------|--->[=]-----(γ'')     ...
               (a)    |
                |    (γ)
                |_____|       ___
...  (x_t-1)-->|__g_t__|-----(x_t)     ...
                               |
                              _|_
                             |f_t|
                               |
                               ⊡
                              y_t

x_t-1   = previous state edge
g_t     = state transition edge
γ       = process noise edge
γp      = previous process noise
γpp     = next process noise
a       = transition coefficient edge
ap      = previous transition coefficient
app     = next transition coefficient
x_t     = current state edge
f_t     = likelihood node
y_t     = observation node
"""

# Start graph
graph = MetaGraph(SimpleGraph(13))

# Previous state edge
set_props!(graph, 1, Dict{Symbol,Any}(:id => "x_tmin", :type => "variable"))

# State transition node
set_props!(graph, 2, Dict{Symbol,Any}(:id => "g_t", :type => "factor"))

# Process noise edge
set_props!(graph, 3, Dict{Symbol,Any}(:id => "γ", :type => "variable"))

# Process noise equality node
set_props!(graph, 4, Dict{Symbol,Any}(:id => "e_γ", :type => "factor"))

# Previous process noise
set_props!(graph, 5, Dict{Symbol,Any}(:id => "γp", :type => "variable"))

# Next process noise
set_props!(graph, 6, Dict{Symbol,Any}(:id => "γpp", :type => "variable"))

# Transition coefficient edge
set_props!(graph, 7, Dict{Symbol,Any}(:id => "a", :type => "variable"))

# Transition coefficient equality node
set_props!(graph, 8, Dict{Symbol,Any}(:id => "e_a", :type => "factor"))

# Previous transition coefficient edge
set_props!(graph, 9, Dict{Symbol,Any}(:id => "ap", :type => "variable"))

# Next transition coefficient edge
set_props!(graph, 10, Dict{Symbol,Any}(:id => "app", :type => "variable"))

# Current state edge
set_props!(graph, 11, Dict{Symbol,Any}(:id => "x_t", :type => "variable"))

# Observation likelihood node
set_props!(graph, 12, Dict{Symbol,Any}(:id => "f_t", :type => "factor"))

# Observation edge
set_props!(graph, 13, Dict{Symbol,Any}(:id => "y_t", :type => "variable"))


add_edge!(graph, 1, 2) # x_t-1 -- g_t
add_edge!(graph, 2, 3) # g_t -- γ
add_edge!(graph, 3, 4) # γ -- e_γ
add_edge!(graph, 4, 5) # e_γ -- γp
add_edge!(graph, 4, 6) # e_γ -- γpp
add_edge!(graph, 2, 7) # g_t -- a
add_edge!(graph, 7, 8) # a -- e_a
add_edge!(graph, 8, 9) # e_a -- ap
add_edge!(graph, 8, 10) # e_a -- app
add_edge!(graph, 2, 11) # g_t -- x_t
add_edge!(graph, 11, 12) # x_t -- f_t
add_edge!(graph, 12, 13) # f_t -- y_t

# Ensure vertices can be recalled from given id
set_indexing_prop!(graph, :id)

"""
Run inference procedure
"""

# Preallocation
estimated_states = zeros(T, 2, TT)
estimated_noises = zeros(T, 2, TT)
estimated_transition = zeros(T, 2, TT)
free_energy_gradients = zeros(T, TT)

# Set state prior x_0
global x_t = EdgeGaussian("x_0"; mean=x0_params[1], precision=x0_params[2])
global app = EdgeGaussian("a_0"; mean=a0_params[1], precision=a0_params[2])
global γpp = EdgeGamma("γ_0"; shape=γ0_params[1], scale=γ0_params[2])

for t = 1:T

      # Report progress
      if mod(t, T/10) == 1
        println("At iteration "*string(t)*"/"*string(T))
      end

      # Previous state
      global x_tmin = EdgeGaussian("x_tmin", mean=x_t.mean, precision=x_t.precision, block=true)

      # Previous transition coefficient
      global ap = EdgeGaussian("ap", mean=app.mean, precision=app.precision, block=true)

      # Previous process noise
      global γp = EdgeGamma("γp", shape=γpp.shape, scale=γpp.scale, block=true)

      # State transition node
      global g_t = TransitionGaussian("g_t", edge_mean="x_tmin", edge_data="x_t", edge_precision="γ", edge_transition="a")

      # Process noise edge
      global γ = EdgeGamma("γ", shape=γpp.shape, scale=γpp.scale)

      # Process noise equality
      global e_γ = NodeEquality("e_γ", edges=["γ", "γp", "γpp"])

      # Next process noise
      global γpp = EdgeGamma("γpp", shape=γpp.shape, scale=γpp.scale, silent=true)

      # Transition coefficient edge
      global a = EdgeGaussian("a", mean=app.mean, precision=app.precision)

      # Transition coefficient equality node
      global e_a = NodeEquality("e_a", edges=["a", "ap", "app"])

      # Next transition coefficient
      global app = EdgeGaussian("app", mean=app.mean, precision=app.precision, silent=true)

      # Current state
      global x_t = EdgeGaussian("x_t", mean=x_tmin.mean, precision=x_tmin.precision)

      # Observation likelihood node
      global f_t = LikelihoodGaussian("f_t", edge_mean="x_t", edge_data="y_t", edge_precision=inv(measurement_noise))

      # Observation edge
      global y_t = EdgeDelta("y_t", observation=observed[t])

      # Start message routine
      act(x_tmin, belief(x_tmin), 1e12, graph);
      act(γpp, belief(γpp), 1e12, graph);
      act(app, belief(app), 1e12, graph);

      # Start clock for reactions
      for tt = 1:TT

        react(g_t, graph)
        react(γ, graph)
        react(e_γ, graph)
        react(γp, graph)
        react(γpp, graph)
        react(a, graph)
        react(e_a, graph)
        react(ap, graph)
        react(app, graph)
        react(x_t, graph)
        react(f_t, graph)
        react(y_t, graph)

        # Write out estimated state parameters
        estimated_states[t, 1, tt] = x_t.mean
        estimated_states[t, 2, tt] = sqrt(1/x_t.precision)
        estimated_noises[t, 1, tt] = γ.shape * γ.scale
        estimated_noises[t, 2, tt] = sqrt(γ.shape * γ.scale^2)
        estimated_transition[t, 1, tt] = a.mean
        estimated_transition[t, 2, tt] = sqrt(1/a.precision)

        # Keep track of FE gradients
        free_energy_gradients[t, tt] = x_t.grad_free_energy
      end
end

"""
Visualize experimental results
"""

# Visualize final state estimates over time
plot(hidden[2:end], color="red", label="states")
plot!(estimated_states[:,1,end], color="blue", label="estimates")
plot!(estimated_states[:,1,end],
      ribbon=[estimated_states[:,2,end], estimated_states[:,2,end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
scatter!(observed, color="black", markersize=3, label="observations")
xlabel!("time (t)")
title!("State estimates, q(x_t)")
savefig(pwd()*"/experiment_infer-coefficients/viz/state_estimates.png")

# Visualize state belief parameter trajectory
t = T
plot(estimated_states[t,1,1:end], color="blue", label="q(x_"*string(t)*")")
plot!(estimated_states[t,1,1:end],
      ribbon=[estimated_states[t,2,1:end], estimated_states[t,2,1:end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
xlabel!("iterations")
title!("Parameter trajectory of q(x_t) for t="*string(t))
savefig(pwd()*"/experiment_infer-coefficients/viz/state_parameter_trajectory_t" * string(t) * ".png")

# Visualize final transition coefficient estimates over time
plot(gain*ones(T,1), color="black", label="transition")
plot!(estimated_transition[:,1,end], color="blue", label="estimates")
plot!(estimated_transition[:,1,end],
      ribbon=[estimated_transition[:,2,end], estimated_transition[:,2,end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
xlabel!("time (t)")
title!("Transition coefficient estimates, q(a)")
savefig(pwd()*"/experiment_infer-coefficients/viz/transition_estimates.png")

# Visualize transition coefficient belief parameter trajectory
t = T
plot(estimated_transition[t,1,1:end], color="blue", label="q(a_"*string(t)*")")
plot!(estimated_transition[t,1,1:end],
      ribbon=[estimated_transition[t,2,1:end], estimated_transition[t,2,1:end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
xlabel!("iterations")
title!("Parameter trajectory of q(a_t) for t="*string(t))
savefig(pwd()*"/experiment_infer-coefficients/viz/transition_parameter_trajectory_t" * string(t) * ".png")

# Visualize final noise estimates over time
plot(inv(process_noise)*ones(T,1), color="black", label="true process noise", linewidth=2)
plot!(estimated_noises[:,1,end], color="blue", label="estimates")
plot!(estimated_noises[:,1,end],
      ribbon=[estimated_noises[:,2,end], estimated_noises[:,2,end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
xlabel!("time (t)")
title!("Noise estimates, q(γ)")
savefig(pwd()*"/experiment_infer-coefficients/viz/noise_estimates.png")

# Visualize noise belief parameter trajectory
t = T
plot(estimated_noises[t,1,1:end], color="blue", label="q(γ_"*string(t)*")")
plot!(estimated_noises[t,1,1:end],
      ribbon=[estimated_noises[t,2,1:end], estimated_noises[t,2,1:end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
xlabel!("iterations")
title!("Parameter trajectory of q(γ) at t="*string(t))
savefig(pwd()*"/experiment_infer-coefficients/viz/noise_parameter_trajectory_t" * string(t) * ".png")

# Visualize free energy gradients over time-series
plot(free_energy_gradients[:,end], color="black", label="||dF||_t")
xlabel!("time (t)")
ylabel!("Norm of free energy gradient")
savefig(pwd()*"/experiment_infer-coefficients/viz/FE_gradients.png")

# Visualize FE gradient for a specific time-step
t = T
plot(free_energy_gradients[t,:], color="blue", label="||dF||_t"*string(t))
savefig(pwd()*"/experiment_infer-coefficients/viz/FE-gradient_trajectory_t" * string(t) * ".png")
