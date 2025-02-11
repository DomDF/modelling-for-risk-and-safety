function get_lognormal_params(μ::Real, σ::Real)
    if μ <= 0
        throw(ArgumentError("mean must be positive"))
    end
    if σ <= 0
        throw(ArgumentError("standard deviation must be positive"))
    end
    
    log_μ = log(μ^2 / sqrt(μ^2 + σ^2))
    log_σ = sqrt(log(1 + (σ^2)/(μ^2)))
    
    return (log_μ = log_μ, log_σ = log_σ)  # named tuple return
end

function add_meas_error(damage::Vector{Float64}, ϵ::Real, seed::Int = 231123)
    if ϵ < 0
        throw(ArgumentError("inspection imprecision must be non-negative"))
    end
    
    return MersenneTwister(seed) |>
        prng -> rand(prng, Normal(0, ϵ), length(damage)) |>
        error -> max.(0, damage + error)
end