## Gradient arithmetic
import Zygote
import Base.+, Base.*, Base.-, Base./

# Add Grads together
function +(G1::Zygote.Grads, G2::Zygote.Grads)
    temp = G1
    for p in G1.params
        temp.grads[p] .+= G2.grads[p]
    end
    return temp
end

# Subtract Grads from each other
function -(G1::Zygote.Grads, G2::Zygote.Grads)
    temp = G1
    for p in G1.params
        temp.grads[p] .-= G2.grads[p]
    end
    return temp
end

# Scalar multiply Grads
function *(G::Zygote.Grads, n::Number)
    temp = G
    for p in G.params
        temp.grads[p] .*= n
    end
    return temp
end

# Scalar divide Grads
function /(G::Zygote.Grads, n::Number)
    temp = G
    for p in G.params
        temp.grads[p] ./= n
    end
    return temp
end
