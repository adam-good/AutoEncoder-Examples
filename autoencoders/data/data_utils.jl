using Flux
using Plots

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

function dataloader(data::DataSample)
    Flux.DataLoader(
        (data.features, Flux.onehotbatch(data.classes, XOR_CLASSES)),
        batchsize=64, shuffle=true
    )
end

function plot_data(features, classes)
    x, y = eachrow(features)
    scatter(x, y, marker_z=classes,
        color=get(cgrad(:heat), classes),
        xlims=(0, 1),
        ylims=(0, 1))
end
