module Xor

using ..Datasets: Dataset, DatasetMetadata, DataSample

export length, features, classes, sample

xor2d(x::Number, y::Number)::Bool = xor(x > 0.5, y > 0.5)
const XOR_CLASS_SYMBOLS = [:False, :True]
const XOR_CLASS_VALUES =  [false, true]
const CLASS_MAP = Dict( #TODO: This is kinda sloppy. Fix
    true => :True, 
    false => :False,
    :True => true,
    :False => false
)

struct XorDataset <: Dataset
    metadata::DatasetMetadata
end
new()::XorDataset = XorDataset(
    DatasetMetadata("XOR", [Float32, Float32], XOR_CLASS_SYMBOLS),
)

struct XorDatasample <: DataSample
    N::Integer
    features::Matrix{Float32}
    classes::Vector{Symbol}
end
length(x::XorDatasample)::Integer = x.N
features(x::XorDatasample)::Matrix{Float32} = x.features
classes(x::XorDatasample)::Vector{Symbol} = x.classes
classes(x::XorDatasample)::Vector{Bool} = map(s -> CLASS_MAP[s], x.classes)

function sample(::XorDataset, N::Integer=1000)::XorDatasample
    data = rand(Float32, 2, N)
    XorDatasample(
        N, data,
        [CLASS_MAP[xor2d(col[1], col[2])]
         for col in eachcol(data)]
    )
end

end
