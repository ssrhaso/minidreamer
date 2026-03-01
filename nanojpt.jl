# nanojpt: dependency-free GPT in Julia. 100 LINE COMPACT NO COMMENT VERSION of microjpt.
using Random, Downloads; Random.seed!(8)
const INPUT = "input.txt"
isfile(INPUT) || Downloads.download("https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt", INPUT)
docs = filter(!isempty, strip.(readlines(INPUT))); shuffle!(docs); println("num docs: $(length(docs))")
const chars = sort(collect(Set(join(docs))))
const BOS = length(chars) + 1; const vocab_size = BOS; println("vocab size: $vocab_size")
encode(s) = [findfirst(==(c), chars) for c in s]
const n_layer, n_embd, block_size, n_head = 4, 64, 16, 8
const hd = n_embd ÷ n_head; init(r, c) = 0.08randn(r, c)
const W = Dict{String,Matrix{Float64}}("wte"=>init(n_embd,vocab_size), "wpe"=>init(n_embd,block_size),
    "lm_head"=>init(vocab_size,n_embd), "attn_wq"=>init(n_embd,n_embd), "attn_wk"=>init(n_embd,n_embd),
    "attn_wv"=>init(n_embd,n_embd), "attn_wo"=>init(n_embd,n_embd),
    "mlp_fc1"=>init(4n_embd,n_embd), "mlp_fc2"=>init(n_embd,4n_embd))
println("num params: $(sum(length, values(W)))")
function rmsnorm(x::Matrix{Float64})
    rr = sqrt.(sum(abs2, x, dims=1) ./ size(x,1) .+ 1e-5); (x ./ rr, rr)
end
rmsnorm_bwd(dy, x, rr) = dy ./ rr .- x .* sum(dy .* x, dims=1) ./ (rr.^3 .* size(x,1))
function fwd(tokens)
    T = length(tokens) - 1; ids = tokens[1:T]
    E = W["wte"][:, ids] .+ W["wpe"][:, 1:T]; x0, r0 = rmsnorm(E)
    xr1 = x0; xn1, r1 = rmsnorm(xr1)
    Q, K, V = W["attn_wq"]*xn1, W["attn_wk"]*xn1, W["attn_wv"]*xn1
    O = similar(Q); As = Vector{Matrix{Float64}}(undef, n_head)
    for h in 1:n_head
        s, e = (h-1)*hd+1, h*hd
        S = K[s:e,:]' * Q[s:e,:] ./ sqrt(hd)
        for t in 1:T, j in t+1:T; S[j,t] = -Inf; end
        S .-= maximum(S, dims=1); eS = exp.(S)
        A = eS ./ sum(eS, dims=1); As[h] = A; O[s:e,:] = V[s:e,:] * A
    end
    xa = W["attn_wo"] * O .+ xr1; xr2 = xa; xn2, r2 = rmsnorm(xr2)
    H = W["mlp_fc1"] * xn2; Hr = max.(0.0, H)
    xm = W["mlp_fc2"] * Hr .+ xr2; lg = W["lm_head"] * xm
    (; ids, T, E, r0, xr1, xn1, r1, Q, K, V, O, As, xr2, xn2, r2, H, Hr, xm), lg
end
function celoss(logits, targets)
    lg = logits .- maximum(logits, dims=1); P = exp.(lg); P ./= sum(P, dims=1)
    T = length(targets); loss = -sum(log(P[targets[t], t]) for t in 1:T) / T
    dl = copy(P); for t in 1:T; dl[targets[t], t] -= 1.0; end; (loss, dl ./ T)
end
function bwd!(cache, dl)
    (; ids, T, E, r0, xr1, xn1, r1, Q, K, V, O, As, xr2, xn2, r2, H, Hr, xm) = cache
    G = Dict(k => zeros(size(v)) for (k,v) in W)
    dx = W["lm_head"]' * dl; G["lm_head"] = dl * xm'; dxr2 = copy(dx)
    dHr = W["mlp_fc2"]' * dx; G["mlp_fc2"] = dx * Hr'
    dH = dHr .* (H .> 0); dxn2 = W["mlp_fc1"]' * dH; G["mlp_fc1"] = dH * xn2'
    dx = rmsnorm_bwd(dxn2, xr2, r2) .+ dxr2; dxr1 = copy(dx)
    dO = W["attn_wo"]' * dx; G["attn_wo"] = dx * O'
    dQ, dK, dV = zeros(n_embd,T), zeros(n_embd,T), zeros(n_embd,T)
    for h in 1:n_head
        s, e = (h-1)*hd+1, h*hd; A = As[h]
        dV[s:e,:] = dO[s:e,:] * A'; dA = V[s:e,:]' * dO[s:e,:]
        dS = A .* (dA .- sum(dA .* A, dims=1))
        dQ[s:e,:] = K[s:e,:] * dS ./ sqrt(hd); dK[s:e,:] = Q[s:e,:] * dS' ./ sqrt(hd)
    end
    G["attn_wq"] = dQ*xn1'; G["attn_wk"] = dK*xn1'; G["attn_wv"] = dV*xn1'
    dxn1 = W["attn_wq"]'*dQ .+ W["attn_wk"]'*dK .+ W["attn_wv"]'*dV
    dx = rmsnorm_bwd(dxn1, xr1, r1) .+ dxr1; dE = rmsnorm_bwd(dx, E, r0)
    for t in 1:T; G["wte"][:,ids[t]] .+= dE[:,t]; G["wpe"][:,t] .+= dE[:,t]; end
    G
end
const lr, β1, β2, ε = 0.01, 0.85, 0.99, 1e-8
const M = Dict(k => zeros(size(v)) for (k,v) in W); const Vb = Dict(k => zeros(size(v)) for (k,v) in W)
function adam_step!(G, step, num_steps)
    lr_t = lr * (1 - (step - 1) / num_steps)
    for k in keys(W)
        M[k] .= β1 .* M[k] .+ (1-β1) .* G[k]; Vb[k] .= β2 .* Vb[k] .+ (1-β2) .* G[k].^2
        W[k] .-= lr_t .* (M[k] ./ (1-β1^step)) ./ (sqrt.(Vb[k] ./ (1-β2^step)) .+ ε)
    end
end
lfmt(x) = (s=string(round(x,digits=4)); d=something(findfirst('.',s),length(s)+1); s*"0"^max(0,4-length(s)+d))
const num_steps = 10000
for step in 1:num_steps
    tokens = [BOS; encode(docs[mod1(step, length(docs))]); BOS][1:min(block_size+1, end)]
    cache, logits = fwd(tokens)
    loss, dl = celoss(logits, tokens[2:end])
    adam_step!(bwd!(cache, dl), step, num_steps)
    print("\rstep $(lpad(step,5)) / $num_steps | loss $(lfmt(loss))")
end
const temperature = 0.5
function infer(n)
    names = String[]
    for _ in 1:n
        toks = [BOS]
        for _ in 1:block_size
            _, lg = fwd(vcat(toks, [BOS]))
            v = lg[:, end] ./ temperature; v .-= maximum(v)
            p = exp.(v); p ./= sum(p); r = rand(); c = 0.0; id = vocab_size
            for i in eachindex(p); c += p[i]; if c >= r; id = i; break; end; end
            id == BOS && break; push!(toks, id)
        end
        push!(names, join(chars[i] for i in toks[2:end]))
    end
    names
end
println("\n INFERENCE (new, hallucinated words): ")
for (i, nm) in enumerate(infer(20)); println("sample $(lpad(i,2)): $nm"); end