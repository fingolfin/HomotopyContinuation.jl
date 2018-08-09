using HomotopyContinuation
using Test
using PolynomialTestSystems
using LinearAlgebra
using Random
import TreeViews

function test_treeviews(x)
    @test TreeViews.hastreeview(x)
    @test_nowarn TreeViews.treelabel(devnull, x, MIME"application/juno+inline"())
    for i=1:TreeViews.numberofnodes(x)
        @test_nowarn TreeViews.nodelabel(devnull, x, i, MIME"application/juno+inline"())
        @test_nowarn TreeViews.treenode(x, i)
    end
end

# We order the tests such that isolated things are tested first
include("utilities_test.jl")
include("multiplicities_test.jl")
include("projective_vectors_test.jl")
include("systems_test.jl")
include("homotopies_test.jl")
include("problem_test.jl")
include("predictors_test.jl")
include("correctors_test.jl")
include("affine_patches_test.jl")
include("path_tracking_test.jl")
include("solve_test.jl")
include("result_test.jl")
include("integration_tests.jl")
