using Distributions, Random, Statistics
using SpecialFunctions

struct LogLogistic{T<:Real} <: ContinuousUnivariateDistribution
    α::T  # scale parameter (alpha)
    β::T  # shape parameter (beta)
    
    function LogLogistic{T}(α::Real, β::Real) where {T<:Real}
        if α <= 0
            throw(ArgumentError("scale parameter α must be positive"))
        end
        if β <= 0
            throw(ArgumentError("shape parameter β must be positive"))
        end
        new{T}(T(α), T(β))
    end
end

# Outer constructor
LogLogistic(α::T, β::T) where {T<:Real} = LogLogistic{T}(α, β)
LogLogistic(α::Real, β::Real) = LogLogistic(promote(α, β)...)

# Required core stats functions
import Statistics: mean, var, std, median, mode, skewness, kurtosis
import Distributions: minimum, maximum, insupport, params, partype

# Parameter accessors
params(d::LogLogistic) = (d.α, d.β)
partype(::LogLogistic{T}) where {T} = T

# Support
minimum(::LogLogistic) = zero(Float64)
maximum(::LogLogistic) = Inf
insupport(::LogLogistic, y::Real) = y >= 0

# PDF matching Stan's formula: (β/α)(y/α)^(β-1) / (1 + (y/α)^β)^2
function pdf(d::LogLogistic, y::Real)
    α, β = params(d)
    if y <= 0
        return zero(y)
    end
    y_over_α = y/α
    return (β/α) * y_over_α^(β-1) / (1 + y_over_α^β)^2
end

# CDF matching Stan's formula: (y/α)^β / (1 + (y/α)^β)
function cdf(d::LogLogistic, y::Real)
    α, β = params(d)
    if y <= 0
        return zero(y)
    end
    y_over_α = y/α
    return y_over_α^β / (1 + y_over_α^β)
end

# Quantile (inverse CDF) matching Stan's RNG formula
function quantile(d::LogLogistic, p::Real)
    if !(0 <= p <= 1)
        throw(ArgumentError("probability must be between 0 and 1"))
    end
    α, β = params(d)
    return α * (p / (1 - p))^(1 / β)
end

# Additional statistics
function mean(d::LogLogistic)
    α, β = params(d)
    if β > 1
        return α * π/β * (1/sin(π/β))
    else
        return Inf
    end
end

function var(d::LogLogistic)
    α, β = params(d)
    if β > 2
        μ = mean(d)
        return α^2 * ((2π/β) * (1/sin(2π/β)) - (π/β)^2 * (1/sin(π/β))^2)
    else
        return Inf
    end
end

std(d::LogLogistic) = sqrt(var(d))

function median(d::LogLogistic)
    α, β = params(d)
    return α  # The scale parameter α is the median
end

function mode(d::LogLogistic)
    α, β = params(d)
    if β > 1
        return α * ((β - 1)/(β + 1))^(1/β)
    else
        return zero(α)
    end
end

# Random number generation
import Random: rand
function rand(rng::AbstractRNG, d::LogLogistic)
    α, β = params(d)
    u = rand(rng)
    return α * (u/(1-u))^(1/β)
end

# Useful additional methods for survival analysis
function survival(d::LogLogistic, y::Real)
    α, β = params(d)
    if y <= 0
        return one(y)
    end
    y_over_α = y/α
    return 1 / (1 + y_over_α^β)
end

function hazard(d::LogLogistic, y::Real)
    return pdf(d, y) / survival(d, y)
end

# Constructor function using Stan's parameter names
function LogLogisticDistribution(α::Real, β::Real)
    LogLogistic(α, β)
end