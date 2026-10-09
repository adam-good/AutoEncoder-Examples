module ModelUtils
using Flux

export probabilities, predict

probabilities(model::Flux.Chain, data) = model(data) |> softmax
predict(probabilities, classes) = probabilities |>
    eachcol .|>
    argmax .|>
    idx -> classes[idx]
end
