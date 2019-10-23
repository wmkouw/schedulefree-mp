# Reactive message-passing in linear Gaussian dynamical system
# Experiment to infer transition coefficients
#
# Wouter Kouw, BIASlab
# 15-08-2019

using Random
using Revise
using Distributions
using DataStructures
using LightGraphs, MetaGraphs
using Plots
pyplot()

# Factor graph components
include("../nodes/node_equality.jl")
include("../nodes/node_gamma.jl")
include("../nodes/node_gaussian.jl")
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
T = 100

# Reaction-time clock
TT = 10

# Known transition and observation matrices
transition_coefficient = 0.8
likelihood_coefficient = 1.0

# Noise parameters
process_noise = 0.5
measurement_noise = 0.3

# Clamped parameters
x0 = [0.1, 1.0]
a0 = [1.0, 1.0]

# Generate data
observed, hidden = gendata_LGDS(transition_coefficient,
                                likelihood_coefficient,
                                process_noise,
                                measurement_noise,
                                x0[1],
                                x0[2],
                                time_horizon=T)

"""
Model/graph specification

This assumes the following type of models
p(y_{1:T}, x_{0:T} | u_{1:T}) = p(x_0) Π_t p(y_t, x_t | x_{t-1})

In other words, Markov chains of time-slices of a state-space models.
Below, we specify the following model through the time-slice subgraph

                e_a
...  (ap)-------[=]------(app)    ...
                _|_
               (a_x)
                 |
                _|_       ___
... (x_t-1)----[g_t]-----(x_t)    ...
                           |
                          _|_
                         [f_t]
                           |
                           ⊡
                          y_t

x_t-1 = previous state edge
g_t   = state transition edge
a_x   = process noise edge
ap    = previous process noise
app   = next process noise
e_a   = process noise equality node
x_t   = current state edge
f_t   = likelihood node
y_t   = observation node
"""

# Start graph
graph = MetaGraph(SimpleGraph(9))

# Edge variable: previous state
set_props!(graph, 1, Dict{Symbol,Any}(:id => "x_tmin", :type => "variable"))

# Factor node: state transition
set_props!(graph, 2, Dict{Symbol,Any}(:id => "g_t", :type => "factor"))

# Edge variable: current transition coefficient
set_props!(graph, 3, Dict{Symbol,Any}(:id => "a_x", :type => "variable"))

# Edge variable: previous transition coefficient
set_props!(graph, 4, Dict{Symbol,Any}(:id => "ap", :type => "variable"))

# Edge variable: next transition coefficient
set_props!(graph, 5, Dict{Symbol,Any}(:id => "app", :type => "variable"))

# Factor node: transition coefficient equality
set_props!(graph, 6, Dict{Symbol,Any}(:id => "e_a", :type => "factor"))

# Edge variable: current state
set_props!(graph, 7, Dict{Symbol,Any}(:id => "x_t", :type => "variable"))

# Factor node: likelihood
set_props!(graph, 8, Dict{Symbol,Any}(:id => "f_t", :type => "factor"))

# Edge variable: observation
set_props!(graph, 9, Dict{Symbol,Any}(:id => "y_t", :type => "variable"))

# Add edges between factor nodes and variables
add_edge!(graph, 1, 2) # x_t-1 -- g_t
add_edge!(graph, 2, 7) # g_t -- x_t
add_edge!(graph, 2, 3) # g_t -- a_x
add_edge!(graph, 3, 6) # a_x -- e_a
add_edge!(graph, 4, 6) # ap -- e_a
add_edge!(graph, 5, 6) # app -- e_a
add_edge!(graph, 7, 8) # x_t -- f_t
add_edge!(graph, 8, 9) # f_t -- y_t

# Ensure vertices can be recalled from given id
set_indexing_prop!(graph, :id)

"""
Run inference procedure
"""

# Preallocation
estimated_states = zeros(T, 2, TT)
estimated_coefficients = zeros(T, 2, TT)
free_energy_gradients = zeros(T, TT)
check = zeros(T, TT)

# Set initial priors
global x_t = EdgeGaussian("x0"; mean=x0[1], precision=x0[2])
global app = EdgeGaussian("a0"; mean=a0[1], precision=a0[2])

# for t = 1:T
t=1

      # Report progress
      if mod(t, T/10) == 1
          println("At iteration "*string(t)*"/"*string(T))
      end

      # Previous state
      global x_tmin = EdgeGaussian("x_tmin", mean=x_t.mean, precision=x_t.precision, block=true)

      # Previous previous noise
      global ap = EdgeGaussian("ap", mean=app.mean, precision=app.precision, block=true)

      # State transition node
      global g_t = TransitionGaussian("g_t", edge_mean="x_tmin", edge_data="x_t", edge_transition="a_x", edge_precision=inv(process_noise))

      # Current process noise
      global a_x = EdgeGaussian("a_x")

      # Transition coefficient equality node
      global e_a = NodeEquality("e_a", edges=["a_x", "ap", "app"])

      # Next transition coefficient
      global app = EdgeGaussian("app", silent=true)

      # Current state
      global x_t = EdgeGaussian("x_t")

      # Observation likelihood node
      global f_t = LikelihoodGaussian("f_t", edge_mean="x_t", edge_data="y_t", edge_precision=inv(measurement_noise))

      # Observation edge
      global y_t = EdgeDelta("y_t", observation=observed[t])

      # Start message routine
      act(x_tmin, belief(x_tmin), 1e3, graph);
      act(ap, belief(ap), 1e3, graph);

      # Start clock for reactions
      for tt = 1:TT

          # Write out estimated state parameters
          estimated_states[t, 1, tt] = x_t.mean
          estimated_states[t, 2, tt] = sqrt(inv(x_t.precision))
          estimated_coefficients[t, 1, tt] = app.mean
          estimated_coefficients[t, 2, tt] = sqrt(inv(app.precision))

          # # Keep track of FE gradients
          # free_energy_gradients[t, tt] = x_t.grad_free_energy

          react(g_t, graph)
          react(a_x, graph)
          react(e_a, graph)
          react(app, graph)
          react(x_t, graph)
          react(f_t, graph)
          react(y_t, graph)
      end
# end

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

# Visualize final noise estimates over time
plot(transition_coefficient*ones(T,1), color="black", label="true process noise", linewidth=2)
plot!(estimated_coefficients[:,1,end], color="blue", label="estimates")
plot!(estimated_coefficients[:,1,end],
      ribbon=[estimated_coefficients[:,2,end], estimated_coefficients[:,2,end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
xlabel!("time (t)")
title!("Coefficient estimates, q(a)")
savefig(pwd()*"/experiment_infer-coefficients/viz/coefficient_estimates.png")

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

# Visualize noise belief parameter trajectory
t = T
plot(estimated_coefficients[t,1,1:end], color="blue", label="q(γ_"*string(t)*")")
plot!(estimated_coefficients[t,1,1:end],
      ribbon=[estimated_coefficients[t,2,1:end], estimated_coefficients[t,2,1:end]],
      linewidth=2,
      color="blue",
      fillalpha=0.2,
      fillcolor="blue",
      label="")
xlabel!("iterations")
title!("Parameter trajectory of q(a) for t="*string(t))
savefig(pwd()*"/experiment_infer-coefficients/viz/coefficient_parameter_trajectory_t" * string(t) * ".png")

# Visualize free energy gradients over time-series
plot(free_energy_gradients[1:end,end], color="black", label="||dF||_t")
xlabel!("time (t)")
ylabel!("Norm of free energy gradient")
savefig(pwd()*"/experiment_infer-coefficients/viz/FE_gradients.png")

# Visualize FE gradient for a specific time-step
t = T
plot(free_energy_gradients[t,:], color="blue", label="||dF||_t"*string(t))
xlabel!("iterations")
ylabel!("Norm of free energy gradient")
savefig(pwd()*"/experiment_infer-coefficients/viz/FE-gradient_trajectory_t" * string(t) * ".png")
