using Coconet
Coconet.train_model(
    # Load your model
    model = nothing,
    opt = nothing,
    losses = nothing,

    # Build model to specs
    head_count = 8,
    latent_size = 32,
    N = 6,

    # Speed it up
    sequence_length = 32,
    batch_size = 5,
    epochs = 20,
    set_length = nothing,
    max_regions = nothing,

    # Difficulty curve
    scramble_rate = 0.1,
    scramble_increase = 0.1,
    scramble_max = 0.9,
    graduation_rate = 0.5,

    # Data
    dataset_name = "Tonnetz_Bach",
    word_size = 3,
)
