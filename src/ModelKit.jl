@reexport module ModelKit

export @var,
    @unique_var,
    Variable,
    Expression,
    coefficients,
    degree,
    degrees,
    differentiate,
    dense_poly,
    evaluate,
    expand,
    exponents_coefficients,
    expressions,
    horner,
    parameters,
    nparameters,
    nvariables,
    monomials,
    parameters,
    subs,
    rand_poly,
    to_number,
    variables,
    System,
    Homotopy

import LinearAlgebra

include("model_kit/symengine.jl")
include("model_kit/symbolic.jl")
include("model_kit/instructions.jl")
include("model_kit/codegen.jl")

end # module
