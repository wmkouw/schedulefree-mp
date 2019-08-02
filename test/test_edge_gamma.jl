using Test

using Distributions: Gamma, mean, params
using DataStructures: Queue, enqueue!, dequeue!
using SpecialFunctions: gamma, digamma

include("../edges/edge_gamma.jl")
include("../nodes/node_gamma.jl")

@testset "constructor" begin

    shape = 1.0
    rate = 1.0
    nodes = Dict{String, Symbol}()
    id = "x"

    x = EdgeGamma(shape, rate, nodes, id)

    @test isa(x.params, Dict{String, Float64})
    @test x.params["shape"] == shape
    @test x.params["rate"] == rate
    @test x.id == id

    # TODO test invalid inputs
end

@testset "update" begin

    shape = 1.0
    rate = 1.0
    nodes = Dict{String, Symbol}()
    id = "x"

    message_left = Gamma(0.5, 2.0)
    message_right = Gamma(1.0, 1.0)

    x = EdgeGamma(shape, rate, nodes, id)
    update(x, message_left, message_right)

    @test x.params["shape"] == 0.5
    @test x.params["rate"] == 3.0

    new_shape, new_rate = params(message(x))
    @test new_shape == 0.5
    @test new_rate == 3.0

    # TODO test invalid inputs
end

@testset "queue_message" begin

    f = NodeGamma(:x, 0.1, 2.0, "f")
    g = NodeGamma(:x, 0.5, 3.0, "g")

    shape = 1.5
    rate = 2.0
    nodes = Dict{String, Symbol}("left" => :f, "right" => :g)
    id = "x"

    message_left = message(f)
    message_right = message(g)

    x = EdgeGamma(shape, rate, nodes, id)
    update(x, message_left, message_right)
    act(x, message(x), 10.0)

    @test dequeue!(f.incoming_messages) == (message(x), 10.0, "outcome")
    @test dequeue!(g.incoming_messages) == (message(x), 10.0, "outcome")

    # TODO test invalid inputs
end
