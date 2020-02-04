module HomotopyContinuation2

export ModelKit

using DoubleFloats: Double64, ComplexDF64
import LinearAlgebra
using Parameters: @unpack
import PrettyTables
import StructArrays

const LA = LinearAlgebra

include("utils.jl")

include("ModelKit.jl")
import .ModelKit

using .ModelKit: @var,
                 @unique_var,
                 Variable,
                 Expression,
                 variables,
                 nvariables,
                 subs,
                 evaluate,
                 differentiate,
                 monomials,
                 expand,
                 System,
                 Homotopy

export @var,
       @unique_var,
       Variable,
       Expression,
       variables,
       nvariables,
       subs,
       evaluate,
       differentiate,
       monomials,
       expand,
       System,
       Homotopy

include("norm.jl")
include("linear_algebra.jl")
include("homotopies.jl")
include("predictors.jl")
include("newton_corrector.jl")
include("tracker.jl")
include("path_info.jl")

end
