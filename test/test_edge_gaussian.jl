using Test

include("../edges/edge_gaussian.jl")

@testset "EdgeGaussian" begin

    x = EdgeGaussian(mean=0., 
                     precision=1.,
                     id="x1",
                     node_l="f1",
                     node_r="f2",
                     node_b="g1")

    @test isa(x.params, Dict{String, Float64})
    @test x.params["mean"] == 0.
    @test x.params["precision"] == 1.
    @test x.edge_id == "x1"
    @test x.node_l == "f1"
    @test x.node_r == "f2"
    @test x.node_b == "g1"
end