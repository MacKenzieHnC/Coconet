module Coconet

# Write your package code here.
include("tonnetz.jl")
include("batching.jl")
include("utils.jl")
include("training.jl")

export pitch_to_tonnetz, tonnetz_to_pitch,
        get_batch, mask,
        +, -, *, /,
        train_model
end
