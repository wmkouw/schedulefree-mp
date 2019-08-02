using Test

include("../nodes/node_gaussian.jl")

@testset "constructor" begin

    f = NodeGaussian("x0", "x1", 1., 1., "f1")

    @test f.node_id == "f1"
    @test f.transition == 1.0
    @test f.precision == 1.0
    @test f.edge_data_id == "x0"
    @test f.edge_mean_id == "x1"

    #TODO: test invalid inputs
end
