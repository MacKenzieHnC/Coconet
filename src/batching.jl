using Flux
using Random

function mask!(region::AbstractArray, scramble_rate)
    masked = zeros(size(region)[1])
    num_masked = Integer(floor(scramble_rate * size(region)[2] * size(region)[3]))
    idx = [(i,j) for i in 1:size(region)[2], j in 1:size(region)[3]]
    idx = shuffle(idx)[1:num_masked]

    for n in idx
        region[:,n...] = masked
    end
    return region
end

function mask(region::AbstractArray, scramble_rate)
    region = deepcopy(region)
    return mask!(region, scramble_rate)
end

function get_batch(dataset, max_sequence::Integer, batch_size::Integer; set::Symbol = :train, max_regions=nothing)
    data = shuffle(dataset[set])

    # Positional data
    word_size = size(data[1])[1]
    region_length = max_sequence
    num_ins = size(data[1])[3]

    # Chop songs up along sequence length
    for j in 1:length(data)
        num_regions = Integer(ceil(size(data[j])[2] / max_sequence))
        if max_regions != nothing
            num_regions = min(num_regions, max_regions)
        end

        data[j] = [data[j][:,max(1, min(i, size(data[j])[2] - max_sequence)):i+min(max_sequence-1, size(data[j])[2]-i-1),:]
                    for i in 1:max_sequence:num_regions*max_sequence]
    end

    # Batch songs
    num_batches = Integer(ceil(length(data) / batch_size))
    data = [data[i:i+min(batch_size-1, length(data)-i)]
                for i in 1:batch_size:batch_size*num_batches]

    return data
end
