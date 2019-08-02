using Test
using Distributions: Normal, params
using DataStructures: Queue, enqueue!, dequeue!

include("../edges/edge_gaussian.jl")
include("../nodes/node_gaussian.jl")

@testset "constructor" begin

    m0 = 0.0
    W0 = 1.0
    state_type = "current"
    nodes = Dict{String, Symbol}()
    id = "x"

    x = EdgeGaussian(m0, W0, state_type, nodes, id)

    @test isa(x.params, Dict{String, Float64})
    @test x.params["mean"] == m0
    @test x.params["precision"] == W0
    @test x.state_type == "current"
    @test x.id == "x"

    # TODO test invalid inputs
end

@testset "update" begin

    m0 = 0.0
    W0 = 1.0
    state_type = "current"
    nodes = Dict{String, Symbol}()
    id = "x"

    message_left = Normal(0.0, 1.0)
    message_right = Normal(1.0, 4.0)

    x = EdgeGaussian(m0, W0, state_type, nodes, id)
    update(x, message_left, message_right)

    @test x.params["precision"] == 5.0
    @test x.params["mean"] == 0.8

    new_mean, new_prec = params(message(x))
    @test new_prec == 5.0
    @test new_mean == 0.8
end

@testset "queue_message" begin

    f = NodeGaussian(:x, :y, 1.0, 1.0, "f")
    g = NodeGaussian(:z, :x, 1.0, 1.0, "g")

    m0 = 0.0
    W0 = 1.0
    state_type = "current"
    nodes = Dict{String, Symbol}("left" => :f, "bottom" => :g)
    id = "x"

    message_left = Normal(0.0, 1.0)
    message_right = Normal(1.0, 4.0)

    x = EdgeGaussian(m0, W0, state_type, nodes, id)
    update(x, message_left, message_right)
    act(x, message(x), 1.0)
    act(x, message(x), 1.0)

    @test dequeue!(f.incoming_messages) == (Normal(0.8, 5.0), 1.0, "data")
    @test dequeue!(g.incoming_messages) == (Normal(0.8, 5.0), 1.0, "mean")
end
