sinusoid(A::Number, ω::Number, ϕ::Number, t::Number) = A*sin(ω*t + ϕ)

struct SinusoidDataset <: Dataset
    metadata::DatasetMetadata
    amplitude::Number
    ang_freq::Number
    phase::Number
end
SinusoidDataset(amplitude, angular_freq, phase)::SinusoidDataset = SinusoidDataset(
DatasetMetadata("Sinusoid", [Float32], [Float32]),
    amplitude, angular_freq, phase
)

struct SinusoidDataSample <: DataSample
    N::Integer
    features::Vector{Float32}
    targets::Vector{Float32}
end
length(x::SinusoidDataSample)::Integer = x.N
features(x::SinusoidDataSample)::Vector{Float32} = x.features
targets(x::SinusoidDataSample)::Vector{Float32} = x.targets

function sample(D::SinusoidDataset, N::Integer=1000)::SinusoidDataSample
    data = rand(Float32, N) .* 2π
    SinusoidDataSample(
        N, data,
        map(t -> sinusoid(D.amplitude, D.ang_freq, D.phase, t), data)
    )
end

export plot_data
function plot_data(data::SinusoidDataSample)
    x,y = features(data), targets(data)
    scatter(x,y, xlims=(0,2π))
end
