using ForwardDiff

print("Testing Partials...")
tic()
include("PartialsTest.jl")
println("done (took $(toq()) seconds).")

print("Testing DiffNumber...")
tic()
include("DiffNumberTest.jl")
println("done (took $(toq()) seconds).")

print("Testing Jacobian-related functionality...")
tic()
include("JacobianTest.jl")
println("done (took $(toq()) seconds).")
