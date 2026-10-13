module Sinusoid

using ..Datasets: Dataset, DatasetMetadata, DataSample

sinusoid(A::Number, ω::Number, ϕ::Number, t::Number) = A*sin(ω*t + ϕ)

struct SinusoidDataset <: Dataset
    metadata::DatasetMetadata
    amplitude::Number
    ang_freq::Number
    phase::Number
end
SinusoidDataset()::SinusoidDataset = SinusoidDataset(
    DatasetMetadata("Sinusoid", [Float32], nothing)
)

struct SinusoidDataSample <: DataSample
    N::Integer,
    features::Matrix{Float32}
    responses::Vector{Float32}
end
length(x::SinusoidDataSample)::Integer = x.N
features(x::SinusoidDataSample)::Matrix{Float32} = x.features
responses(x::SinusoidDataSample)::Vector{Float32} = x.responses

function sample(D::SinusoidDataset, N::Integer=1000)::SinusoidDataSample
    data = rand(Float32, N)
    SinusoidDataSample(
        N, data,
        map(t -> sinusoid(D.amplitude, D.ang_freq, D.phase, t), data)
    )
end

end
