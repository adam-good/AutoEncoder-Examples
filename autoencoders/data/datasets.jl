module Datasets

struct DatasetMetadata
    Name::String
    feature_types::Vector{Type}
    classes::Vector{Symbol}
end

abstract type DataSample end
length(::DataSample) = error("DataSample Subtypes Must Implement length()")
features(::DataSample) = error("DataSample Subtypes Must Implement features()")
classes(::DataSample) = error("DataSample Subtypes Must Implement classes()")
#ith_feature(i::Integer, x::DataSample) = eachcol(x.features)[i]
#ith_class(i::Integer, x::DataSample) = x.classes[i]

abstract type Dataset end
sample(::Dataset, ::Any)::DataSample = error("Dataset Subtypes Must Implement sample()")

include("datasets/xor.jl")

end
