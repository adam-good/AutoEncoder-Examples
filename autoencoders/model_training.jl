module ModelTraining

include("models/models.jl")

using Flux
using Flux.Optimisers
using Flux.Losses
using ProgressMeter

using .Models

const DEFAULT_OPT = Optimisers.Adam(0.01)
struct ModelTrainer{M,S}
    model::M
    opt_state::S
end
ModelTrainer(model::Flux.Chain, opt::Optimisers.AbstractRule=DEFAULT_OPT) = ModelTrainer(model, Flux.setup(opt, model))
model(trainer::ModelTrainer)::Flux.Chain = trainer.model

function update!(trainer, x, y)
    loss, grads = Flux.withgradient(trainer.model) do m
        Losses.logitcrossentropy(m(x), y)
    end
    Flux.update!(trainer.opt_state, trainer.model, grads[1])
    loss
end

function train!(trainer, dataloader, epochs)
    losses = []
    @showprogress for _ in 1:epochs
        for (x, y) in dataloader
            loss = update!(trainer, x, y)
            push!(losses, loss)
        end
    end
end

function validate(model, x, y; metrics::Dict{Symbol,Function}=MLMetrics.DEFAULT_METRICS)
    y_pred = model(x) |> softmax |> predict
    conf_mat = MLMetrics.confusion_matrix(y_pred, y)

    Dict(metric => fn(conf_mat) for (metric, fn) = metrics)

end

end
