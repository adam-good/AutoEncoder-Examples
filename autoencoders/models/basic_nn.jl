using Flux: Losses
using Flux
using ProgressMeter

struct BasicNN <: Model
    in_size::Int
    hidden_sizes::Vector{Int}
    out_size::Int
    inner::Flux.Chain
end
function BasicNN(n_features::Int, hidden_sizes::Vector{Int}, n_targets::Int)::BasicNN
    layers = Vector{Any}()
    push!(layers, Dense(n_features => hidden_sizes[1], relu))
    for i = 1:(length(hidden_sizes)-1)
        push!(layers, Dense(hidden_sizes[i] => hidden_sizes[i+1], relu))
    end
    push!(layers, Dense(hidden_sizes[end] => n_targets))

    model = Flux.Chain(layers...)

    BasicNN(n_targets, hidden_sizes, n_targets, model)
end
BasicNN(n_features::Int, n_targets::Int)::BasicNN = BasicNN(n_features, [n_features+n_targets], n_targets) 
get_trainable(m::BasicNN) = m.inner
(nn::BasicNN)(x::AbstractMatrix) = nn.inner(x)

function update!(network::Flux.Chain, opt_state, x, y)
    loss, grads = Flux.withgradient(network) do m 
        Losses.logitcrossentropy(m(x), y)
    end
    Flux.update!(opt_state, network, grads[1])
    loss
end
update!(model::BasicNN, opt_state, x, y) = update!(get_trainable(model), opt_state, x, y)

function train!(network::Flux.Chain, opt_state, dataloader, epochs)
    losses = []
    @showprogress for _ in 1:epochs
        counter = 0
        batch_loss = 0
        for (x,y) in dataloader
            loss = update!(network, opt_state, x, y)
            batch_loss += loss
            counter += 1
        end
        batch_loss /= counter
        push!(losses, batch_loss)
    end
    losses
end
train!(model::BasicNN, opt_state, x, y) = train!(get_trainable(model), opt_state, x, y)
