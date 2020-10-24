using Coconet
using Flux
using BSON
using Plots
using MusicTransformer

## Hyper Parameters

function train_model(;
    model = nothing,
    opt = nothing,
    losses = nothing,

    head_count = 8,
    latent_size = 32,
    word_size = 3,
    N = 6,

    sequence_length = 32,
    batch_size = 5,
    epochs = 1,
    set_length = nothing,
    max_regions = nothing,

    scramble_rate = 0.1,
    scramble_increase = 0.1,
    scramble_max = 0.9,
    graduation_rate = 0.5,

    dataset_name = "Tonnetz_Bach",
)

    difficulty_level = 1
    difficulties = [scramble_rate:scramble_increase:scramble_max...]

    pos_data = nothing

    ## Get model
    epoch_start = 1

    if model == nothing
        @info "Building model..."
        model = Transformer(N, head_count, word_size, latent_size)
        opt = Flux.ADAM()
    end

    model = model |> gpu
    ps = params(model)

    ## Data
    @info "Wrangling data"
    path = "datasets"
    dataset = BSON.load(joinpath(path, string(dataset_name, ".bson")))

    if set_length != nothing
        dataset[:train] = dataset[:train][1:set_length]
        dataset[:test] = dataset[:test][1:set_length]
    end

    ## Loss
    function loss(query_data, y)
        l = y .- model(query_data, pos_data)
        l = l .* l
        sum(l)
    end

    function test_loss(set::Symbol)
        L_test = 0 # Total loss
        i = 0 # counter for the progress bar

        test = Coconet.get_batch(
            dataset,
            sequence_length,
            batch_size,
            set = set,
            max_regions = max_regions,
        )

        for batch in test
            L_batch = 0 # Loss per batch
            for song in batch
                L_song = 0 # loss per song
                song = gpu(song)
                for region in song
                    # data
                    pos_data = gpu(get_positional_encoding(region))
                    query_data = Coconet.mask(region, scramble_rate)
                    key_data = query_data
                    y = region

                    # Loss per region
                    L_song += loss(query_data, y)
                end

                # Average of song losses across batch
                L_batch += L_song

                # Progress bar
                if i % 4 == 0
                    if i % 16 == 0
                        print(' ')
                    end
                    print('/')
                end
                i += 1
            end

            # Losses across test set
            L_test += L_batch
        end

        print(string("> ", L_test, "\n"))
        return L_test
    end

    function ratio()
        @info string("Current Ratio")
        print(string("Train: ", losses[length(losses)][1]/losses[1][1], "\n"))
        print(string("Test: ", losses[length(losses)][2]/losses[1][2], "\n"))
    end

    ## Plot
    function save_to_plot(epoch)
        train = [loss[1] for loss in losses] ./ losses[1][1]
        test = [loss[2] for loss in losses] ./ losses[1][2]
        x = 0:1:(length(losses)-1)


        p = plot(x, train, label="Train")
        plot!(x, test, label="Test")

        ylims!(0, max(train..., test..., 1))
        xlims!(0, length(losses) - 1)

        xlabel!("Epochs")
        p = ylabel!("percent of loss_0")

        p = plot(
            p,
            size = (1280,720),
            xticks = x,
            yticks = 0:0.1:1,
        )

        savefig(p, string("checkpoint/loss_curve"))
        return p
    end

    ## TRAINING

    # Get initial loss
    @info string("Initial Loss:\n")
    if losses == nothing
        print("Train: ")
        tr_l = test_loss(:train)

        print("Test: ")
        te_l = test_loss(:test)

        losses = [(tr_l, te_l)]
    else
        print(string("Train: ", losses[1][1],"\n"))
        print(string("Test: ", losses[1][2],"\n"))
    end
    ratio()
    best_loss = losses[length(losses)][2]
    last_improvement = best_loss / losses[1][2]

    # Train!
    for epoch = epoch_start:(epoch_start+epochs - 1)
        i = 0 # counter for the progress bar
        @info string("Epoch ", epoch)
        train = Coconet.get_batch(
            dataset,
            sequence_length,
            batch_size,
            set = :train,
            max_regions = max_regions,
        )

        L_train = 0
        batch_num = 1

        # Epoch
        for batch in train
            print(string("Batch ", batch_num," of ",length(train)))

            ∇_batch = [] # Gradient per batch
            L_batch = 0 # Loss per batch
            for song in batch
                ∇_song = [] # Gradient per song
                L_song = 0 # Loss per song
                song = gpu(song)
                for region in song
                    # Data
                    query_data = Coconet.mask(region, scramble_rate)
                    key_data = query_data
                    y = region
                    pos_data = gpu(get_positional_encoding(region))

                    L_region = 0

                    # Calculate gradient over region
                    ∇_region = gradient(ps) do
                        L_region = loss(query_data, y)
                    end

                    push!(∇_song, cpu(∇_region))
                    L_song += L_region
                end

                # Store song gradients
                push!(∇_batch, ∇_song)
                L_batch += L_song

                # Progress bar
                print('.')
            end
            print("\n")

            ∇_batch = gpu(∇_batch)
            # Combine gradients
            ∇_batch = sum([sum(∇_song) for ∇_song in ∇_batch])
            L_train += L_batch

            # Apply gradients (a.k.a. Train!)
            Flux.update!(opt, ps, ∇_batch)
            batch_num += 1
        end

        # Get loss
        L_test = test_loss(:test)
        push!(losses, (L_train, L_test))

        # Print current state
        ratio()

        #FUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU
        if isnan(L_test)
            print("FUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU")
            break
        end

        # Make it harder if it's doing too well
        test_ratio = losses[length(losses)][2] / losses[1][2]
        improvement = 1 - (test_ratio / last_improvement)

        if improvement > graduation_rate
            if difficulty_level < length(difficulties)
                difficulty_level += 1
                last_improvement = test_ratio
                @info string("Increasing scramble rate to ", difficulties[difficulty_level])
            end
        elseif improvement < 0
            last_improvment = test_ratio
            @warn string("Looks like they're struggling...")
        end

        # Save checkpoint
        if losses[length(losses)][2] < best_loss
            @info "Saving model"
            best_loss = losses[length(losses)][2]
            m = model |> cpu
            o = opt |> cpu
            BSON.@save string("checkpoint/my_model.bson") model=m opt=o epoch_start=epoch+1 losses=losses scramble_rate
        end

        # Plot losses
        save_to_plot(epoch)
    end
end
