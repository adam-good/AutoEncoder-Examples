module Xor

using ..Datasets: Dataset, DatasetMetadata, DataSample

export length, features, classes, sample

xor2d(x::Number, y::Number)::Bool = xor(x > 0.5, y > 0.5)
const XOR_CLASSES = [:False, :True]
const CLASS_MAP = Dict(true => :True, false => :False)

struct XorDataset <: Dataset
    metadata::DatasetMetadata
end
new()::XorDataset = XorDataset(
    DatasetMetadata("XOR", [Float32, Float32], XOR_CLASSES),
)

struct XorDatasample <: DataSample
    N::Integer
    features::Matrix{Float32}
    classes::Vector{Symbol}
end
length(x::XorDatasample) = x.N
features(x::XorDatasample) = x.features
classes(x::XorDatasample) = x.classes

function sample(::XorDataset, N::Integer=1000)::XorDatasample
    data = rand(Float32, 2, N)
    XorDatasample(
        N, data,
        [CLASS_MAP[xor2d(col[1], col[2])]
         for col in eachcol(data)]
    )
end

end
