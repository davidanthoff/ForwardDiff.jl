########################
# Math with GradNumVec #
########################

## Addition/Subtraction ##
##----------------------##

+{N,A,B}(z::GradNumVec{N,A}, w::GradNumVec{N,B}) = GradNumVec{N,promote_type(A,B)}(value(z)+value(w), partials(z)+partials(w))
+{N,T}(x::Real, z::GradNumVec{N,T}) = GradNumVec{N,promote_type(typeof(x),T)}(x+value(z), partials(z))
+{N,T}(z::GradNumVec{N,T}, x::Real) = x+z

-{N,T}(z::GradNumVec{N,T}) = GradNumVec{N,T}(-value(z), -partials(z))
-{N,A,B}(z::GradNumVec{N,A}, w::GradNumVec{N,B}) = GradNumVec{N,promote_type(A,B)}(value(z)-value(w), partials(z)-partials(w))
-{N,T}(x::Real, z::GradNumVec{N,T}) = GradNumVec{N,promote_type(typeof(x),T)}(x-value(z), -partials(z))
-{N,T}(z::GradNumVec{N,T}, x::Real) = GradNumVec{N,promote_type(T,typeof(x))}(value(z)-x, partials(z))

## Multiplication ##
##----------------##

# avoid ambiguous definition with Bool*Number
*{N,T}(x::Bool, z::GradNumVec{N,T}) = ifelse(x, z, ifelse(signbit(value(z))==0, zero(z), -zero(z)))
*{N,T}(z::GradNumVec{N,T}, x::Bool) = x*z

function *{N,A,B}(z::GradNumVec{N,A}, w::GradNumVec{N,B})
    z_r, w_r = value(z), value(w)
    T = promote_type(A,B)
    dus = mul_dus!(Array(T, N), partials(z), partials(w), z_r, w_r)
    return GradNumVec{N,T}(z_r*w_r, dus)
end

function mul_dus!(result, zdus, wdus, z_r, w_r)
    @simd for i=1:length(result)
        @inbounds result[i] = (zdus[i] * w_r) + (z_r * wdus[i])
    end
    return result
end

function *{N,A}(x::Real, z::GradNumVec{N,A})
    T = promote_type(typeof(x),T)
    return GradNumVec{N,T}(x*value(z), x*partials(z))
end

*{N,T}(z::GradNumVec{N,T}, x::Real) = x*z

## Division ##
##----------##

/{N,A,B<:Real}(z::GradNumVec{N,A}, x::B) = GradNumVec{N,promote_type(A,B,Float64)}(value(z)/x, partials(z)/x)

function /{N,A<:Real,B}(x::A, z::GradNumVec{N,B})
    z_r = value(z)
    T = promote_type(A, B, Float64)
    dus = div_real_by_dus!(Array(T, N), -x, partials(z), z_r^2)
    return GradNumVec{N,T}(x/z_r, dus)
end

function div_real_by_dus!(result, neg_x, dus, z_r_sq)
    @simd for i=1:length(result)
        @inbounds result[i] = (neg_x * dus[i]) / z_r_sq
    end
    return result
end

function /{N,A,B}(z::GradNumVec{N,A}, w::GradNumVec{N,B})
    z_r, w_r = value(z), value(w)
    T = promote_type(A, B, Float64)
    dus = div_dus!(Array(T,N), partials(z), partials(w), z_r, w_r, w_r^2)
    return GradNumVec{N,T}(z_r/w_r, dus)
end

function div_dus!(result, zdus, wdus, z_r, w_r, denom)
    @simd for i=1:length(result)
        @inbounds result[i] = ((zdus[i] * w_r) - (z_r * wdus[i]))/denom
    end
    return result
end

## Exponentiation ##
##----------------##

for f in (:^, :(NaNMath.pow))

    @eval function ($f){N,A,B}(z::GradNumVec{N,A}, w::GradNumVec{N,B})
        z_r, w_r = value(z), value(w)    
        re = $f(z_r, w_r)
        powval = w_r * (($f)(z_r, w_r-1))
        logval = ($f)(z_r, w_r) * log(z_r)
        T = promote_type(A,B)
        dus = mul_dus!(Array(T, N), partials(z), partials(w), logval, powval)
        return GradNumVec{N,T}(re, dus)
    end

    @eval ($f)(::Base.MathConst{:e}, z::GradNumVec) = exp(z)

    # generate redundant definitions to resolve ambiguity warnings
    for R in (:Integer, :Rational, :Real)
        @eval function ($f){N,A}(z::GradNumVec{N,A}, x::$R)
            z_r = value(z)
            powval = x*($f)(z_r, x-1)
            T = promote_type(A,typeof(x))
            return GradNumVec{N,T}(($f)(z_r, x), powval*partials(z))
        end

        @eval function ($f){N,A}(x::$R, z::GradNumVec{N,A})
            z_r = value(z)
            logval = ($f)(x, z_r)*log(x)
            T = promote_type(typeof(x), A)
            return GradNumVec{N,T}(($f)(x, z_r), logval*partials(z))
        end
    end

end

## from Calculus.jl ##
##------------------##

for (funsym, ex) in Calculus.symbolic_derivatives_1arg()
    funsym == :exp && continue
    
    @eval function $(funsym){N,A}(z::GradNumVec{N,A})
        x = value(z) # `x` is the variable name for $exp
        df = $ex
        T = promote_type(A, typeof(df), Float64)
        return GradNumVec{N,T}($(funsym)(x), df*partials(z))
    end

    # extend corresponding NaNMath methods
    if funsym in (:sin, :cos, :tan,
                  :asin, :acos, :acosh,
                  :atanh, :log, :log2,
                  :log10, :lgamma, :log1p)

        nan_funsym = Expr(:.,:NaNMath,Base.Meta.quot(funsym))

        @eval function $(nan_funsym){N,A}(z::GradNumVec{N,A})
            x = value(z) # `x` is the variable name for $ex
            df = $(to_nanmath(ex))
            T = promote_type(A, typeof(df), Float64)
            return GradNumVec{N,T}($(nan_funsym)(x), df*partials(z))
        end
    end
end

function exp{N,A}(z::GradNumVec{N,A})
    df = exp(value(z))
    T = promote_type(A, typeof(df), Float64)
    return GradNumVec{N,T}(df, df*partials(z))
end