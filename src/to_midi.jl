using MIDI
using BSON
using Flux
using CUDA
using MusicTransformer
using Coconet

tpq = 960
sixteenth = tpq / 4

BSON.@load "checkpoint/my_model.bson" model

dataset = BSON.load("datasets/Tonnetz_Bach.bson")

model = gpu(model)


test_song = gpu(dataset[:valid][rand(1:length(dataset[:valid]))])
song_mask = Coconet.mask(test_song, 0.9) |> gpu
masked_song = test_song ./ (song_mask[1,1,:,:,:] .+ 1)
pos_data = gpu(get_positional_encoding(test_song))

prediction = model(masked_song, pos_data, mask = song_mask)

loss = test_song .- prediction
loss = sum(loss .* loss)

song = zeros(size(prediction)[2:end])

# for each instrument
for i in 1:size(prediction, 3)
    for j in 1:size(prediction, 2)
        if round(prediction[3,j,i]) > 0
            pitch = tonnetz_to_pitch(prediction[1,j,i], prediction[2,j,i])
            song[j,i] = pitch
        end
    end
end

song

file = MIDIFile(1, tpq, output_tracks)
writeMIDIFile(string("results/output_file_", 1), file)
