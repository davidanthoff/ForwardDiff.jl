function dual_fad{T <: Real}(f::Function, x::Vector{T}, gradient_output, dualvec)
  @assert(length(dualvec) == length(x),
          "The length of x ($(length(x))) is different from n ($(length(dualvec))).")
  for i in 1:length(x)
    dualvec[i] = Dual(x[i], zero(T))
  end
  for i in 1:length(x)
    dualvec[i] = Dual(real(dualvec[i]), one(T))
    gradient_output[i] = epsilon(f(dualvec))
    dualvec[i] = Dual(real(dualvec[i]), zero(T))
  end
end

function dual_fad_gradient!(f::Function, ::Type{Float64}; n::Int=1)
  dualvec = Array(Dual{Float64}, n)
  g!(x::Vector{Float64}, gradient_output::Vector{Float64}) = dual_fad(f, x, gradient_output, dualvec)
  return g!
end

function dual_fad_gradient(f::Function, ::Type{Float64}; n::Int=1)
  dualvec = Array(Dual{Float64}, n)
  gradient_output = Array(Float64, n)
  function g(x::Vector{Float64})
    dual_fad(f, x, gradient_output, dualvec)
    gradient_output
  end
  return g
end
