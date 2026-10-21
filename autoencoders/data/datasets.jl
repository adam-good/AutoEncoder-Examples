module Datasets

export length, features, targets, sample

struct DatasetMetadata
    Name::String
    feature_types::Vector{Type}
    target_types::Vector{Type}
end

abstract type DataSample end
length(::DataSample) = error("DataSample Subtypes Must Implement length()")
features(::DataSample) = error("DataSample Subtypes Must Implement features()")
targets(::DataSample) = error("DataSample Subtypes Must Implement targets()")
#ith_feature(i::Integer, x::DataSample) = eachcol(x.features)[i]
#ith_class(i::Integer, x::DataSample) = x.classes[i]

abstract type Dataset end
function sample end
sample(::Dataset, ::Any)::DataSample = error("Dataset Subtypes Must Implement sample()")

include("datasets/xor.jl")
include("datasets/sinusoid.jl")

end
