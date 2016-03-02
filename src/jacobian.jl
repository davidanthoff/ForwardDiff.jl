########################
# @jacobian!/@jacobian #
########################

const JACOBIAN_KWARG_ORDER = (:allresults, :chunk, :xlength, :multithread)

macro jacobian!(args...)
    args, kwargs = separate_kwargs(args)
    arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS, JACOBIAN_KWARG_ORDER)
    return esc(:(ForwardDiff._jacobian!($(args...), $(arranged_kwargs...))))
end

macro jacobian(args...)
    args, kwargs = separate_kwargs(args)
    arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS, JACOBIAN_KWARG_ORDER)
    return esc(:(ForwardDiff._jacobian($(args...), $(arranged_kwargs...))))
end

##################
# JacobianResult #
##################

immutable JacobianResult{L,T} <: ForwardDiffResult
    xlength::Val{L}
    raw::T
end

function jacobian{L}(result::JacobianResult{L})
    out = similar(result.raw, numtype(eltype(result.raw)), length(result.raw), L)
    return jacobian!(out, result)
end

function jacobian!{L}(out, result::JacobianResult{L})
    ylength = length(result.raw)
    @assert size(out) == (ylength, L)
    for j in 1:L
        @simd for i in 1:ylength
            @inbounds out[i, j] = partials(result.raw[i], j)
        end
    end
    return out
end

function value(result::JacobianResult)
    out = similar(result.raw, numtype(eltype(result.raw)))
    return jacobian!(out, result)
end

function value!(out, result::JacobianResult)
    @assert length(out) == length(result.raw)
    @simd for i in 1:length(result.raw)
        @inbounds out[i] = value(result.raw[i])
    end
    return out
end

########################
# _jacobian!/_jacobian #
########################

function _jacobian!{A,C,L,M}(out, f!, y, x, allresults::Val{A}, chunk::Val{C}, xlength::Val{L}, multithread::Val{M})
    result = call_jacobian!(multithread, pickchunk(chunk, xlength, x), f!, y, x)
    value!(y, result)
    jacobian!(out, result)
    return pickresult(allresults, result, out)
end

function _jacobian!{A,C,L,M}(out, f, x, allresults::Val{A}, chunk::Val{C}, xlength::Val{L}, multithread::Val{M})
    result = call_jacobian!(multithread, pickchunk(chunk, xlength, x), f, x)
    jacobian!(out, result)
    return pickresult(allresults, result, out)
end

function _jacobian{A,C,L,M}(f!, y, x, allresults::Val{A}, chunk::Val{C}, xlength::Val{L}, multithread::Val{M})
    result = call_jacobian!(multithread, pickchunk(chunk, xlength, x), f!, y, x)
    value!(y, result)
    out = jacobian(result)
    return pickresult(allresults, result, out)
end

function _jacobian{A,C,L,M}(f, x, allresults::Val{A}, chunk::Val{C}, xlength::Val{L}, multithread::Val{M})
    result = call_jacobian!(multithread, pickchunk(chunk, xlength, x), f, x)
    out = jacobian(result)
    return pickresult(allresults, result, out)
end

#######################
# workhorse functions #
#######################

@inline function call_jacobian!{N}(multithread, ::Tuple{Val{N}, Val{N}}, args...)
    return vector_mode_jacobian!(Val{N}(), args...)
end

@inline function call_jacobian!{C,L}(::Val{false}, ::Tuple{Val{C}, Val{L}}, args...)
    return chunk_mode_jacobian!(Val{C}(), Val{L}(), args...)
end

@inline function call_jacobian!{C,L}(::Val{true}, ::Tuple{Val{C}, Val{L}}, args...)
    return multi_chunk_mode_jacobian!(Val{C}(), Val{L}(), args...)
end

###############
# vector mode #
###############

# J(f(x)) #
#---------#

function vector_mode_jacobian!{L}(xlength::Val{L}, f, x)
    diffx = fetchdiffx(x, xlength)
    seeds = fetchseeds(diffx)
    seed!(diffx, x, seeds)
    return JacobianResult(xlength, f(diffx))
end

# J(f!(y, x)) #
#-------------#

function vector_mode_jacobian!{L}(xlength::Val{L}, f!, y, x)
    diffx = fetchdiffx(x, xlength)
    diffy = Vector{DiffNumber{L,eltype(y)}}(length(y))
    seeds = fetchseeds(diffx)
    seed!(diffx, x, seeds)
    f!(diffy, diffx)
    return JacobianResult(xlength, diffy)
end

##############
# chunk mode #
##############

# J(f(x)) #
#---------#

function chunk_mode_jacobian!{chunk,xlength}(::Val{chunk}, ::Val{xlength}, f, x)
    # TODO
end

# J(f!(y, x)) #
#-------------#

function chunk_mode_jacobian!{chunk,xlength}(::Val{chunk}, ::Val{xlength}, f!, y, x)
    # TODO
end

# @generated function _jacobian_chunk_mode!{chunk, input_length}(f, outarg, x, ::Type{Val{chunk}}, ::Type{Val{input_length}})
#     if outarg <: DummyOutput
#         outputdef = :(output = Matrix{S}(output_length, input_length))
#     else
#         outputdef = quote
#             @assert size(outarg) == (output_length, input_length)
#             output = outarg
#         end
#     end
#     remainder = input_length % chunk == 0 ? chunk : input_length % chunk
#     fill_length = input_length - remainder
#     reseed_partials = remainder == chunk ? :() : :(seed_partials = cachefetch!(tid, Partials{chunk,T}, Val{$(remainder)}))
#     return quote
#         @assert input_length == length(x)
#         T = eltype(x)
#         tid = compat_threadid()
#         zero_partials = zero(Partials{chunk,T})
#         seed_partials = cachefetch!(tid, Partials{chunk,T})
#         workvec = cachefetch!(tid, DiffNumber{chunk,T}, Val{input_length})
#
#         # do the first chunk manually, so that we can infer the dimensions
#         # of the output matrix if necessary
#         @simd for i in 1:input_length
#             @inbounds workvec[i] = DiffNumber{chunk,T}(x[i], zero_partials)
#         end
#         @simd for i in 1:chunk
#             @inbounds workvec[i] = DiffNumber{chunk,T}(x[i], seed_partials[i])
#         end
#         chunk_result = f(workvec)
#         S, output_length = numtype(eltype(chunk_result)), length(chunk_result)
#         $(outputdef)
#         for i in 1:chunk
#             @simd for r in 1:output_length
#                 @inbounds output[r, i] = partials(chunk_result[r], i)
#             end
#             @inbounds workvec[i] = DiffNumber{chunk,T}(x[i], zero_partials)
#         end
#
#         # now do the rest of the chunks until we hit the fill_length
#         for c in $(chunk + 1):$(chunk):$(fill_length)
#             offset = c - 1
#             @simd for i in 1:chunk
#                 j = i + offset
#                 @inbounds workvec[j] = DiffNumber{chunk,T}(x[j], seed_partials[i])
#             end
#             chunk_result = f(workvec)
#             for i in 1:chunk
#                 j = i + offset
#                 @simd for r in 1:output_length
#                     @inbounds output[r, j] = partials(chunk_result[r], i)
#                 end
#                 @inbounds workvec[j] = DiffNumber{chunk,T}(x[j], zero_partials)
#             end
#         end
#
#         # do the final remaining chunk manually
#         $(reseed_partials)
#         @simd for i in 1:$(remainder)
#             j = $(fill_length) + i
#             @inbounds workvec[j] = DiffNumber{chunk,T}(x[j], seed_partials[i])
#         end
#         chunk_result = f(workvec)
#         @simd for i in 1:$(remainder)
#             j = $(fill_length) + i
#             @simd for r in 1:output_length
#                 @inbounds output[r, j] = partials(chunk_result[r], i)
#             end
#         end
#
#         return JacobianResult(chunk_result, output)
#     end
# end
