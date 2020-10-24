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

scaling_factor = Coconet.unit_matrix
scaling_inverse = inv(scaling_factor)

# Load in as dict
bach = BSON.load(joinpath(path, name))

#=  Problem 1: Finding the data structure
        Songs can all be of different length, so a song will be
            about the biggest thing we can fit in a single matrix
        So I think a whole dataset will be
            a vector of vectors of song matrices with size
            pitch x instrument x timestep (or some other order,
            if we decide we like it better for some reason)
        And save the whole thing as a bson
        Then for actual batching, we can scramble instead of silence
            the notes we want to mask, so we can hopefully not make
            our algorithm quite so neurotic about silence, but hopefully
            also have a sense of when a note is "wrong" so that
            we're free to do some weird stuff with encoding silences.

    Problem 2: Encoding notes
        The structure for a specific pitch of a specific instrument
            at a specific timestep (pitch, instrument, timestep)
            will be a 3-dimensional vector of floats
            The first two parts of the vector are the x and y coordinates
            of a pitch in tonnetz space
            And the third is a float between 0 and 1 representing if the
            instrument is playing (1 for yes, 0 for no),
            (see "tonnetz.jl" for in-depth explanation)

        If an instrument is currently playing, just encode its
            position (x,y, velocity) in tonnetz-space

    Problem 2: Encoding silences
        What I think I want to do, is have the silent instrument(s) sort of
        hum the other parts to themselves.
=#

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
            end # for each timestep
        end # for each instrument

        push!(set, song)
    end #for each song

    push!(dataset, set)
end # for each set

#===============================================================================
    SECTION 3: Save it
===============================================================================#
bson(
    joinpath(path, name),
    test = dataset[1],
    train = dataset[2],
    valid = dataset[3],
    scaling_factor = scaling_factor,
    scaling_inverse = scaling_inverse,
)
