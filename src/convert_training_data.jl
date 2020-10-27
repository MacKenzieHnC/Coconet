## Convert orignal Coconet training data into Julia format
using BSON
using PyCall
using Coconet

#===============================================================================
    SECTION 1: Python to Julia
===============================================================================#

path = "datasets"
name = "Tonnetz_Bach.bson"

let np = pyimport("numpy")
    # Load data as PyObject
    data = np.load(
        joinpath(path, "Jsb16thSeparated.npz"),
        allow_pickle = true,
        encoding = "latin1",
    )

    # Convert to Julia vectors
    test_data = [[timestep for timestep in song] for song in get(data, :test)]
    train_data = [[timestep for timestep in song] for song in get(data, :train)]
    valid_data = [[timestep for timestep in song] for song in get(data, :valid)]

    # Convert to arrays
    test_data = [hcat(song...) for song in test_data]
    train_data = [hcat(song...) for song in train_data]
    valid_data = [hcat(song...) for song in valid_data]

    # Transpose and re-convert to arrays
    test_data = [hcat(song') for song in test_data]
    train_data = [hcat(song') for song in train_data]
    valid_data = [hcat(song') for song in valid_data]

    # Save as dict
    bson(
        joinpath(path, name),
        test = test_data,
        train = train_data,
        valid = valid_data,
    )
end
#===============================================================================
    SECTION 2: Notes to tonnetz vectors
===============================================================================#

# Load in as dict
bach = BSON.load(joinpath(path, name))

notes = []
dataset = []
# for each component of the dataset
for test in [bach[v] for v in [:test, :train, :valid]]
    set = []

    # for each song
    for i = 1:length(test)
        song = zeros(Float32, 3, size(test[i])...)

        # for each instrument
        for instrument = 1:size(test[i])[2]
            # for each timestep
            for timestep = 1:size(test[i])[1]
                pitch = test[i][timestep, instrument]
                x, y, z = 0, 0, 0

                if !isnan(pitch)
                    z = 1
                    x, y = Coconet.pitch_to_tonnetz(Integer(pitch))
                end

                song[:, timestep, instrument] = [x,y,z]
                push!(notes, song[:, timestep, instrument])
            end # for each timestep
        end # for each instrument

        push!(set, song)
    end #for each song

    push!(dataset, set)
end # for each set

# Normalization
μ = sum(notes) / length(notes)
σ = sum([(note - μ).^2 for note in notes]) / length(notes)
for i in 1:length(dataset)
    for j in 1:length(dataset[i])
        for k in 1:size(dataset[i][j], 2)
            for l in 1:size(dataset[i][j], 3)
                dataset[i][j][:, k, l] = (dataset[i][j][:, k, l] - μ) ./ σ
            end
        end
    end
end

#===============================================================================
    SECTION 3: Save it
===============================================================================#
bson(
    joinpath(path, name),
    test = dataset[1],
    train = dataset[2],
    valid = dataset[3],
    μ = μ,
    σ = σ,
)
