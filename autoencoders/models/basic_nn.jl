module Models

using Flux

# TODO: Add some parameters as args so this is more flexible
function new_basic_nn()
    Flux.Chain(
        Dense(2 => 3, tanh),
        BatchNorm(3),
        Dense(3 => 2)
    )
end

end
