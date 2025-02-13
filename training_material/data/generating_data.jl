using Distributions, Random # for generating data
using CSV # for reading and writing files
using DataFrames, DataFramesMeta, CategoricalArrays # for data manipulation

include("useful_functions.jl")
include("LogLogisticDistribution.jl")

cd(@__DIR__)

#########################################
#
# corrosion growth rate
#
#########################################

n_anomalies = 25; n_soil_A = 20

initial_damage = get_lognormal_params(0.1, 0.05) |>
    params -> LogNormal(params.log_μ, params.log_σ)

soil_a_cgr = Normal(0.1, 0.05) |> d -> truncated(d, lower = 0)
soil_b_cgr = Normal(0.01, 0.05) |> d -> truncated(d, lower = 0)

initial_damage_samples = MersenneTwister(231123) |> prng -> rand(prng, initial_damage, n_anomalies)
soil_a_cgr_samples = MersenneTwister(231123) |> prng -> rand(prng, soil_a_comp, n_soil_A)
soil_b_cgr_samples = MersenneTwister(231123) |> prng -> rand(prng, soil_b_comp, n_anomalies - n_soil_A)

cgr_samples = vcat(soil_a_cgr_samples, soil_b_cgr_samples)

Δt = [5, 10]

damage_two = initial_damage_samples .+ cgr_samples * Δt[1]
damage_three = initial_damage_samples .+ cgr_samples * Δt[2]

inspection_imprecision = 0.05

DataFrame(
    anomaly_id = 1:n_anomalies,
    soil_type = [repeat(["A"], n_soil_A); repeat(["B"], n_anomalies - n_soil_A)],
    i_0 = add_meas_error(initial_damage_samples, inspection_imprecision, 1),
    i_1 = add_meas_error(damage_two, inspection_imprecision, 2),
    i_2 = add_meas_error(damage_three, inspection_imprecision, 3)
    ) |>
    df -> stack(df, 
                [:i_0, :i_1, :i_2], 
                variable_name = :inspection, value_name = :measured_depth_mm) |>
    df -> @transform(df, :T = replace(:inspection, "i_0" => 0, "i_1" => 5, "i_2" => 10)) |>
    data -> CSV.write("corrosion.csv", data)

CSV.read("corrosion.csv", DataFrame) |>
    df -> scatterplot(df.T, df.measured_depth_mm, 
                     title = "Corrosion depth vs time",
                     xlabel = "Time (years)",
                     ylabel = "Depth (mm)")
                     
#########################################
#
# probability of indication
#
#########################################

# Set up parameters
n_inspections = 100

# True parameters for logistic regression
β₀ = -6        # intercept
β_depth = 4/5    # depth coefficient
β_length = 1/5   # length coefficient
PFI = 0.05     # probability of false indication

# Generate damage sizes from realistic distributions
test_depths = LinRange(1, 10, 10)
test_lengths = LinRange(3, 30, 10)

DataFrame(
    anomaly_id = 1:n_inspections,
    depth_mm = [d for (d, l) in Iterators.product(test_depths, test_lengths)] |> vec,
    length_mm = [l for (d, l) in Iterators.product(test_depths, test_lengths)] |> vec
    ) |>
    df -> @rtransform(df, 
        :indication = PoI.(β₀ .+ β_depth .* :depth_mm .+ β_length .* :length_mm, PFI) |>
            poi -> rand(MersenneTwister(:anomaly_id), Bernoulli(poi))
        ) |>
    data -> CSV.write("poi.csv", data)


CSV.read("poi.csv", DataFrame) |>
    df -> scatterplot(df.depth_mm, df.indication,
                     title = "Probability of indication vs depth",
                     xlabel = "Depth (mm)",
                     ylabel = "Indication?")  


#########################################
#
# survival analysis
#
#########################################

# True parameter values - chosen to be realistic for industrial equipment
true_α = 10    # scale parameter (median lifetime)
true_β = 6    # shape parameter (increasing hazard rate)

# Simulation parameters
n_samples = 20              # number of components to simulate
inspection_interval = 1.5    # time between inspections
max_time = 12              # maximum observation time

# Generate true failure times
d = LogLogisticDistribution(true_α, true_β)
true_failures = rand(d, n_samples)

# Create inspection times
inspection_times = 0:inspection_interval:max_time

# Function to find the bracketing inspection times
function find_inspection_bounds(failure_time, inspection_times)
    if failure_time > maximum(inspection_times)
        return (maximum(inspection_times), Inf)
    end
    
    for (t1, t2) in zip(inspection_times[1:end-1], inspection_times[2:end])
        if t1 <= failure_time < t2
            return (t1, t2)
        end
    end
    
    return (0.0, inspection_times[1])
end

# Create DataFrame with the simulated data
DataFrame(
    component_id = 1:n_samples,
    true_failure_time = true_failures
) |>
    df -> @rtransform(df,
        :bounds = find_inspection_bounds(:true_failure_time, inspection_times)) |>
    df -> @rtransform(df,
        :fail_lb = first(:bounds),
        :fail_ub = last(:bounds)
        ) |>
    df -> @select(df, :component_id, :fail_lb, :fail_ub) |>
    df -> CSV.write("failures.csv", df)
