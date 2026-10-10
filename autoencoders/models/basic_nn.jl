module BasicNN

using Flux

using ..Models: Model

struct BasicNN <: Model end
# TODO: Add some parameters as args so this is more flexible
new()::BasicNN = Flux.Chain(
        Dense(2 => 3, tanh),
        BatchNorm(3),
        Dense(3 => 2)
    )

end
