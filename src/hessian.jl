######################
# @hessian!/@hessian #
######################

macro hessian!(args...)
    args, kwargs = separate_kwargs(args)
    arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS)
    return esc(:(ForwardDiff.hessian_entry_point!($(arranged_kwargs...), $(last(args)), $(args[1:end-1]...))))
end

macro hessian(args...)
    args, kwargs = separate_kwargs(args)
    arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS)
    return esc(:(ForwardDiff.hessian_entry_point($(arranged_kwargs...), $(last(args)), $(args[1:end-1]...))))
end

##################
# HessianResult #
##################

abstract HessianResult <: ForwardDiffResult

# vector mode #
#-------------#

immutable HessianVectorResult{D} <: HessianResult
    len::Int
    dual::D
end

function hessian(result::HessianVectorResult)
    out = Matrix{numtype(numtype(result.dual))}(result.len, result.len)
    return hessian!(out, result)
end

function hessian!(out, result::HessianVectorResult)
    @assert size(out) == (result.len, result.len)
    for j in 1:result.len
        @simd for i in 1:result.len
            @inbounds out[i, j] = partials(partials(result.dual, i), j)
        end
    end
    return out
end

function gradient(result::HessianVectorResult)
    out = Vector{numtype(numtype(result.dual))}(result.len)
    return gradient!(out, result)
end

function gradient!(out, result::HessianVectorResult)
    @assert length(out) == result.len
    dval = value(result.dual)
    @simd for i in 1:result.len
        @inbounds out[i] = partials(dval, i)
    end
    return out
end

value(result::HessianVectorResult) = value(value(result.dual))

# chunk mode #
#------------#

immutable HessianChunkResult{T,G,H} <: HessianResult
    value::T
    grad::G
    hess::H
end

hessian(result::HessianChunkResult) = result.hess

hessian!(out, result::HessianChunkResult) = copy!(out, result.hess)

gradient(result::HessianChunkResult) = result.grad

gradient!(out, result::HessianChunkResult) = copy!(out, result.grad)

value(result::HessianChunkResult) = result.value

###############
# API methods #
###############

function hessian_entry_point!(chunk, len, allresults, multithread, x, args...)
    return dispatch_hessian!(pickchunk(chunk, len, x), allresults, multithread, x, args...)
end

function hessian_entry_point(chunk, len, allresults, multithread, x, args...)
    return dispatch_hessian(pickchunk(chunk, len, x), allresults, multithread, x, args...)
end

# vector mode #
#-------------#

@inline function dispatch_hessian!{N}(::Tuple{Val{N}, Val{N}}, allresults, multithread, x, out, f)
    result = vector_mode_hessian!(Val{N}(), f, x)
    hessian!(out, result)
    return pickresult(allresults, result, out)
end

@inline function dispatch_hessian{N}(::Tuple{Val{N}, Val{N}}, allresults, multithread, x, f)
    result = vector_mode_hessian!(Val{N}(), f, x)
    out = hessian(result)
    return pickresult(allresults, result, out)
end

# chunk mode #
#------------#

@inline function dispatch_hessian!{C,L}(::Tuple{Val{C}, Val{L}}, allresults, multithread, x, out, f)
    result = chunk_mode_hessian!(multithread, Val{C}(), Val{L}(), out, f, x)
    return pickresult(allresults, result, out)
end

@inline function dispatch_hessian{C,L}(::Tuple{Val{C}, Val{L}}, allresults, multithread, x, f)
    result = chunk_mode_hessian!(multithread, Val{C}(), Val{L}(), DummyVar(), f, x)
    return pickresult(allresults, result, result.grad)
end

#######################
# workhorse functions #
#######################

# vector mode #
#-------------#

function vector_mode_hessian!{L}(len::Val{L}, f, x)
    @assert length(x) == L
    xdual = fetchxdual(x, len, len, len)
    inseeds = fetchseeds(numtype(eltype(xdual)))
    outseeds = fetchseeds(eltype(xdual))
    seed!(xdual, x, 1, inseeds, outseeds)
    return HessianVectorResult(L, f(xdual))
end

# chunk mode #
#------------#

@generated function chunk_mode_hessian!{C,L}(multithread::Val{false}, chunk::Val{C}, len::Val{L}, outvar, f, x)
    if outvar <: DummyVar
        outdef = :(out = Matrix{numtype(eltype(outdual))}(L, L))
    else
        outdef = quote
            @assert size(outvar) == (L, L)
            out = outvar
        end
    end
    lastchunksize = L % C == 0 ? C : L % C
    fullchunks = div(L - lastchunksize, C)
    lastoffset = L - lastchunksize + 1
    reseedexpr = lastchunksize == C ? :() : :(seeds = fetchseeds(eltype(xdual), $(Val{lastchunksize}())))
    return quote
        @assert length(x) == L
        xdual = fetchxdual(x, len, chunk, chunk)
        inseeds = fetchseeds(numtype(eltype(xdual)))
        outseeds = fetchseeds(eltype(xdual))
        inzeroseed = zero(Partials{C,numtype(eltype(xdual))})
        outzeroseed = zero(Partials{C,eltype(xdual)})
        seedall!(xdual, x, len, inzeroseed, outzeroseed)

        # do first chunk manually
        seed!(xdual, x, 1, seeds)
        dual = f(xdual)
        seed!(xdual, x, 1, zeroseed)
        outdual
        hessloadchunk!(out, dual, 1, chunk)

        # do middle chunks
        for c in 2:$(fullchunks)
            offset = ((c - 1) * C + 1)
            seed!(xdual, x, offset, seeds)
            dual = f(xdual)
            seed!(xdual, x, offset, zeroseed)
            hessloadchunk!(out, dual, chunk, offset)
        end

        # do final chunk manually
        $(reseedexpr)
        seed!(xdual, x, $(lastoffset), seeds)
        dual = f(xdual)
        hessloadchunk!(out, dual, $(lastoffset), $(Val{lastchunksize}()))

        $(outdef)


        return HessianChunkResult(L, outdual, out)
    end
end
#
#
# function hessloadchunk!{C}(out, dual, offset, chunk::Val{C})
#     k = offset - 1
#     for i in 1:C
#         j = i + k
#         out[j] = partials(dual, i)
#     end
# end
