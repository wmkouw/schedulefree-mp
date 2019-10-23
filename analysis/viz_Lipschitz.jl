# Small script to visualize Lipschitz constants in various forms
#
# Wouter Kouw
# 21-10-2019

using LinearAlgebra
using Plots
pyplot()

# Function
f(x) = sin.(x)
Df(x) = cos.(x)

# Choose d
d = 1.

# Range
x = collect(range(-2*pi,stop=2*pi, step=0.01))
y = x .+ d

# Plot
plot(x, f(x), color="red", label="f(x)")
plot!(x, f(y), color="blue", label="f(x+d)")
plot!(x, norm.(Df(y) - Df(x), 2), color="black", label="|| ∇f(x+d) - ∇f(x)||_2")

# println("For f(x)=sin(x) and d=1.0, L < 1.0")
