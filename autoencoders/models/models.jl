module Models

export Model, BasicNN, probabilities, predict

using Flux

include("metrics.jl")
using .Metrics

abstract type Model end
(::Model)(::AbstractVector) = error("Model must be implemented for each subtype")

probabilities(model::Model, data) = model(data) |> softmax
predict(probabilities, classes) = begin
    probabilities |>
        eachcol .|>
        argmax .|>
        idx -> classes[idx]
end

include("basic_nn.jl")

end
