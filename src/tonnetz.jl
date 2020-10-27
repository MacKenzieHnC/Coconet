using Coconet
#=
    The core idea here is that we can derive Tonnetz very simply,
    as a small repeating tile.

    We start with a small number game, where we start from 0,
    and increase by 4 when going to the right, and 1 when going up

     ⋮  ⋮  ⋮  ⋮
    4  8 12 16 ...
    3  7 11 15 ...
    2  6 10 14 ...
    1  5  9 13 ...
    0  4  8 12 ...

    in other words, the value n of a position (x, y) can be expressed as
        n = 4x + y

    You may also notice some repetition
        starting with the 4s and going to the right.
        In fact, we can tile this pattern infinitely.

         ⋮  ⋮  ⋮  ⋮
    ... 4  8 12 16 ...
    ... 3  7 11 15 ...
    ... 2  6 10 14 ...
    ... 1  5  9 13 ...
    ... 0  4  8 12 16 ...
       ... 3  7 11 15 ...
       ... 2  6 10 14 ...
       ... 1  5  9 13 ...
       ... 0  4  8 12 ...
          ⋮   ⋮  ⋮  ⋮

    Now, since its infinitely repeating, it's not possible to have an inverse,
        but we can fix that too.
    If we just restrict our space to the first 4 rows, we can densely describe
        the infinite space pretty simply, while working our way toward
        an invertible function.

    We can describe a point's position x, y as
        x = floor(n / 4) <-- we only want integer values
        y = n % 4

    This grid, is the only way we will be interacting with notes.
    Any changes to the space will be handled by multiplying our results by
        a matrix representing a linear transformation.

    The last thing we need to do is add a y_offset, so that we can
        reach any point in the grid, which gives

        x = floor(n / 4)
        y = (n % 4) + y_offset

    We now have a geometric interpretation of all possible notes.
        You could even use negative values if you have an idea of what a
        "negative pitch" could possibly mean.
=#

#= Convert a pitch into a position in the tonnetz grid

        Pitch is just the numeric pitch (0-127 inclusive)
        y_offset lets you control where in the infinite
            grid of possible spaces you want to be.
            I intend to use it to center on the last note played,
            thus sort of minimizing the distance between consecutive notes
            in tonnetz space
=#
function pitch_to_tonnetz(pitch::Integer; y_offset = 0::Integer)

    # Subtract the vertical offset so pitches aren't changed
    pitch = pitch - y_offset

    # Since columns increase by 4,
    #   the x component is just the pitch divided by 4 with no remainder, and
    #   the y component is just the remainder
    x = floor(pitch / 4)
    y = pitch % 4

    #= Julia allows negative solutions to the mod function,
        e.g. -1 % 4 = -1
        but we want -1 % 4 = 3,
        So we have to manually correct for that =#
    y = (y < 0) ? 4 + y : y

    #= All our numbers are now correct,
    But our space is still anchored to (0, 0),
    So add the offset to y to get the correct location =#
    y = y + y_offset

    # We now have an infinite, navigatable grid of pitches
    #   that are related spacially to the tonnetz
    return x, y
end

# Convert a position on the tonnetz grid back to a pitch
function tonnetz_to_pitch(x, y)

    # Choose it based on its grid position
    pitch = x * 4 + y

    # Ta-da
    return Integer(round(pitch))
end

# Convert a song of tonnetz into an array of pitches
function song_to_pitch_array(song::AbstractArray)
    pitch_size = size(song, 1)
    sequence_length = size(song, 2)
    num_instruments = size(song, 3)

    output = zeros(sequence_length, num_instruments)

    for i in 1:num_instruments
        for j in 1:sequence_length
            pitch = NaN
            if round(song[3, j, i]) > 0
                pitch = Coconet.tonnetz_to_pitch(song[1:2, j, i])
            end

            output[i, j] = pitch
        end
    end
end
