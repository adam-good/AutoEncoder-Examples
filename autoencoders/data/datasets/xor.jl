
using Plots
using Flux

xor2d(x::Number, y::Number)::Bool = xor(x > 0.5, y > 0.5)
const XOR_CLASS_SYMBOLS = Set([:False, :True])
const XOR_CLASS_VALUES = [false, true]
const CLASS_MAP = Dict( #TODO: This is kinda sloppy. Fix
    true => :True,
    false => :False,
    :True => true,
    :False => false
)

struct XorDataset <: Dataset
    metadata::DatasetMetadata
end
XorDataset()::XorDataset = XorDataset(
    DatasetMetadata("XOR", [Float32, Float32], [Symbol]),
)

struct XorDatasample <: DataSample
    N::Integer
    features::Matrix{Float32}
    targets::Vector{Symbol}
end
length(x::XorDatasample)::Integer = x.N
features(x::XorDatasample)::Matrix{Float32} = x.features
targets(x::XorDatasample)::Vector{Symbol} = x.targets
targets(x::XorDatasample)::Vector{Bool} = map(s -> CLASS_MAP[s], x.targets)

function sample(::XorDataset, N::Integer=1000)::XorDatasample
    data = rand(Float32, 2, N)
    XorDatasample(
        N, data,
        [CLASS_MAP[xor2d(col[1], col[2])]
         for col in eachcol(data)]
    )
end

function DataLoader(data::XorDatasample)::Flux.DataLoader
    Flux.DataLoader(
        (data.features, Flux.onehotbatch(data.targets, XOR_CLASS_SYMBOLS)),
        batchsize=64, shuffle=true
    )
end

function plot_data(data::XorDatasample, z=nothing)
    color = z === nothing ? nothing : get(cgrad(:heat), z)
    x, y = eachrow(features(data))
    scatter(x, y, marker_z=z,
        color=color,
        xlims=(0, 1),
        ylims=(0, 1))
end
