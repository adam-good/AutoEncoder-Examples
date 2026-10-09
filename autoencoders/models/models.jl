module Models

using Flux

include("basic_nn.jl")
include("metrics.jl")

using .Metrics

probabilities(model::Flux.Chain, data) = model(data) |> softmax
predict(probabilities, classes) = begin
    probabilities |>
    eachcol .|>
    argmax .|>
    idx -> classes[idx]
end

end
