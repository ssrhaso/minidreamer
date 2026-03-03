""" 
microgpt: A first-principles GPT in ~100 lines of pure Julia.
A complete neural architecture distilled to its absolute minimum:
encoder -> cross-attention state -> decoder.

Authored by @ssrhaso. Inspired by MicroGPT by @karpathy. 
"""

# PROLOGUE : The Setup 
using Random, Downloads; Random.seed!(8)                                     

# ACT I : The Data & Parameters 
#   SCENE 1 : The English Words Dataset
const INPUT = "input.txt"                                                   # Define local file path
isfile(INPUT) || Downloads.download("https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt", INPUT)       # If no file, download
docs = filter(!isempty, strip.(readlines(INPUT))); shuffle!(docs);          # Read file, strip whitespace, and randomize
println("num docs: $(length(docs))")                                        # Announce the size of our training corpus
const chars = sort(collect(Set(join(docs))))                                # Extract all unique characters to build our alphabet
const BOS = length(chars) + 1; const vocab_size = BOS                       # Total vocabulary size is characters + BOS token
println("vocab size: $vocab_size")                                          # Announce input dimensionality
encode(s) = [findfirst(==(c), chars) for c in s]                            # The Tokenizer: mapping human language to integers

# SCENE 2 : The Initialization of Parameters
const n_layer, n_embd, block_size, n_head = 4, 64, 16, 8                    # Hyperparameters: depth, width, context size, attention heads
const hd = n_embd ÷ n_head; init(r, c) = 0.08randn(r, c)                    # Derived dimension per head, and Gaussian WEIGHT initialization, σ = 0.08
const W = Dict{String,Matrix{Float64}}(                                     # Allocate the global dictionary of weight matrices
    "wte"=>init(n_embd,vocab_size), "wpe"=>init(n_embd,block_size),         # Token and Positional embeddings (64x28 and 64x16)
    "lm_head"=>init(vocab_size,n_embd), "attn_wq"=>init(n_embd,n_embd),     # Final linear projection and Attention Queries (28x64 and 64x64)
    "attn_wk"=>init(n_embd,n_embd), "attn_wv"=>init(n_embd,n_embd),         # Attention Keys and Values (64x64 and 64x64)
    "attn_wo"=>init(n_embd,n_embd), "mlp_fc1"=>init(4n_embd,n_embd),        # Attention Output projection and 1st MLP layer (64x64 and 256x64)
    "mlp_fc2"=>init(n_embd,4n_embd))                                        # 2nd MLP layer (64x256)
println("num params: $(sum(length, values(W)))")                            # Announce the total mathematical complexity

# ACT II : The Forward Pass 
#   SCENE 1 : Normalization
# RMSNorm stabilizes the variance of our hidden states.
function rmsnorm(x::Matrix{Float64})                        
    rr = sqrt.(sum(abs2, x, dims=1) ./ size(x,1) .+ 1e-5); (x ./ rr, rr)    # Calculate RMS variance for each column; return norm and scale
end
rmsnorm_bwd(dy, x, rr) = dy ./ rr .- x .* sum(dy .* x, dims=1) ./ (rr.^3 .* size(x,1)) # The exact derivative of RMSNorm

#   SCENE 2 : The Forward Projection
# No scalar loops here; Julia breathes pure matrix math.
function fwd(tokens)                                        
    T = length(tokens) - 1; ids = tokens[1:T]                               # Extract sequence length and the input token IDs 
    E = W["wte"][:, ids] .+ W["wpe"][:, 1:T]; x0, r0 = rmsnorm(E)           # Pluck and combine token and position embeddings, then normalize
    
    xr1 = x0; xn1, r1 = rmsnorm(xr1)                                        # Save residual connection, then normalize input for Attention
    Q, K, V = W["attn_wq"]*xn1, W["attn_wk"]*xn1, W["attn_wv"]*xn1          # Project input into Multi-head Attention Queries, Keys, Values
    O = similar(Q); As = Vector{Matrix{Float64}}(undef, n_head)             # Allocate memory for the attention output and weights
    
    for h in 1:n_head                                                       # Iterate through each independent attention head
        s, e = (h-1)*hd+1, h*hd                                             # Calculate slice indices for this specific head
        S = K[s:e,:]' * Q[s:e,:] ./ sqrt(hd)                                # Compute raw attention scores: Key transpose dot Query
        for t in 1:T, j in t+1:T; S[j,t] = -Inf; end                        # Apply Causal Mask: override future tokens with negative infinity
        S .-= maximum(S, dims=1); eS = exp.(S)                              # Shift scores for stability, then exponentiate
        A = eS ./ sum(eS, dims=1); As[h] = A; O[s:e,:] = V[s:e,:] * A       # Normalize to sum to 1 (Softmax), then multiply by Values
    end
    
    xa = W["attn_wo"] * O .+ xr1; xr2 = xa; xn2, r2 = rmsnorm(xr2)          # Project combined heads, add 1st residual, save 2nd residual, normalize
    H = W["mlp_fc1"] * xn2; Hr = max.(0.0, H)                               # Project through MLP expansion layer, apply ReLU non-linearity
    xm = W["mlp_fc2"] * Hr .+ xr2; lg = W["lm_head"] * xm                   # Project through MLP compression, add residual, project to logits
    (; ids, T, E, r0, xr1, xn1, r1, Q, K, V, O, As, xr2, xn2, r2, H, Hr, xm), lg # Return the massive activation cache, and the final logits
end

# ACT III : The Loss and Manual Matrix Backpropagation 
#   SCENE 1 : The Tragedy of Cross-Entropy
# Define Cross-Entropy Loss to measure how wrong our predictions are.
function celoss(logits, targets)                            
    lg = logits .- maximum(logits, dims=1); P = exp.(lg); P ./= sum(P, dims=1) # Shift logits, convert to probability distribution (Softmax)
    T = length(targets); loss = -sum(log(P[targets[t], t]) for t in 1:T) / T   # Calculate negative log likelihood of the correct target tokens
    dl = copy(P); for t in 1:T; dl[targets[t], t] -= 1.0; end; (loss, dl ./ T) # Subtract 1 from correct target to complete Softmax derivative
end

#   SCENE 2 : Manual Backpropagation: The Matrix Calculus
# No autograd required. We manually push gradients backward using raw matrix calculus.
function bwd!(cache, dl)                                    
    (; ids, T, E, r0, xr1, xn1, r1, Q, K, V, O, As, xr2, xn2, r2, H, Hr, xm) = cache # Unpack all saved forward activations
    G = Dict(k => zeros(size(v)) for (k,v) in W)                            # Allocate a dictionary of zero-matrices to accumulate gradients
    
    dx = W["lm_head"]' * dl; G["lm_head"] = dl * xm'; dxr2 = copy(dx)       # Gradient of lm_head input and weights; branch for MLP residual
    dHr = W["mlp_fc2"]' * dx; G["mlp_fc2"] = dx * Hr'                       # Gradient propagating back through 2nd MLP layer and weights
    dH = dHr .* (H .> 0); dxn2 = W["mlp_fc1"]' * dH; G["mlp_fc1"] = dH * xn2' # ReLU backward; backprop through 1st MLP layer and weights
    dx = rmsnorm_bwd(dxn2, xr2, r2) .+ dxr2; dxr1 = copy(dx)                # Backprop through MLP RMSNorm, recombine with residual branch
    
    dO = W["attn_wo"]' * dx; G["attn_wo"] = dx * O'                         # Gradient propagating back through Attention output projection
    dQ, dK, dV = zeros(n_embd,T), zeros(n_embd,T), zeros(n_embd,T)          # Allocate gradients for Queries, Keys, and Values
    for h in 1:n_head                                                       # Iterate backwards through the attention heads
        s, e = (h-1)*hd+1, h*hd; A = As[h]                                  # Calculate slice indices; retrieve cached attention weights
        dV[s:e,:] = dO[s:e,:] * A'; dA = V[s:e,:]' * dO[s:e,:]              # Derivative of Values; Derivative of Attention weights
        dS = A .* (dA .- sum(dA .* A, dims=1))                              # The Softmax Jacobian: complex chain rule for normalized exponents
        dQ[s:e,:] = K[s:e,:] * dS ./ sqrt(hd); dK[s:e,:] = Q[s:e,:] * dS' ./ sqrt(hd) # Derivative of Queries and Keys
    end
    
    G["attn_wq"] = dQ*xn1'; G["attn_wk"] = dK*xn1'; G["attn_wv"] = dV*xn1'  # Accumulate gradients for Q, K, and V weight matrices
    dxn1 = W["attn_wq"]'*dQ .+ W["attn_wk"]'*dK .+ W["attn_wv"]'*dV         # Combine gradients flowing back from Q, K, and V
    dx = rmsnorm_bwd(dxn1, xr1, r1) .+ dxr1; dE = rmsnorm_bwd(dx, E, r0)    # Backprop through Attention and initial RMSNorms, add residual
    for t in 1:T; G["wte"][:,ids[t]] .+= dE[:,t]; G["wpe"][:,t] .+= dE[:,t]; end # Iterate over sequence to scatter gradients to embeddings
    G                                                                       # Return the completed dictionary of gradients
end

# ACT IV : The Sacred Adam Optimizer and Training Loop 
#   SCENE 1 : The Optimizer
const lr, β1, β2, ε = 0.01, 0.85, 0.99, 1e-8                                # Define Learning rate, momentum, RMSprop, and safety epsilon
const M = Dict(k => zeros(size(v)) for (k,v) in W)                          # Allocate 1st moment buffer (momentum of gradients)
const Vb = Dict(k => zeros(size(v)) for (k,v) in W)                         # Allocate 2nd moment buffer (variance of gradients)
function adam_step!(G, step, num_steps)                                     
    lr_t = lr * (1 - (step - 1) / num_steps)                                # Compute linear learning rate decay 
    for k in keys(W)                                                        # Iterate over every weight matrix in the model
        M[k] .= β1 .* M[k] .+ (1-β1) .* G[k]; Vb[k] .= β2 .* Vb[k] .+ (1-β2) .* G[k].^2 # Update moving averages of gradients and squared gradients
        W[k] .-= lr_t .* (M[k] ./ (1-β1^step)) ./ (sqrt.(Vb[k] ./ (1-β2^step)) .+ ε)    # Apply bias correction and update the actual weights
    end
end

#   SCENE 2 : The Grand Rehearsal
lfmt(x) = (s=string(round(x,digits=4)); d=something(findfirst('.',s),length(s)+1); s*"0"^max(0,4-length(s)+d)) 
const num_steps = 10000                                                     # Define total number of training iterations
for step in 1:num_steps                                                     
    tokens = [BOS; encode(docs[mod1(step, length(docs))]); BOS]             # Grab document, tokenize it, and sandwich between BOS tokens
    tokens = tokens[1:min(block_size+1, end)]                               # Truncate sequence to fit within our attention context window
    cache, logits = fwd(tokens)                                             # Project forward: predict the next tokens
    loss, dl = celoss(logits, tokens[2:end])                                # Calculate how wrong we were, and generate initial loss gradient
    adam_step!(bwd!(cache, dl), step, num_steps)                            # Unravel backprop, get weight gradients, and update parameters
    print("\rstep $(lpad(step,5)) / $num_steps | loss $(lfmt(loss))")       # Overwrite terminal line to show real-time progress
end

# FINALE : Hallucination and Inference 
#   SCENE 1 : The Model "Speaks"
# Temperature controls the creative spark. May the model dream back to us.
const temperature = 0.5                                                     # Define how random the sampling is (lower = safer, higher = chaotic)
function infer(n)                                                           # Function to generate n hallucinated words
    names = String[]                                                        
    for _ in 1:n                                                            # Loop to generate 'n' independent words
        toks = [BOS]                                                        # Start sequence with blank slate (Beginning of Sequence)
        for _ in 1:block_size                                               # Generate token by token up to the context limit
            _, lg = fwd(vcat(toks, [BOS]))                                  # Forward pass current sequence
            v = lg[:, end] ./ temperature; v .-= maximum(v)                 # Pluck final logit column, apply temperature, shift for stability
            p = exp.(v); p ./= sum(p); r = rand(); c = 0.0; id = vocab_size # Convert to probability distribution via Softmax
            for i in eachindex(p); c += p[i]; if c >= r; id = i; break; end; end # Sample from distribution using cumulative threshold
            id == BOS && break; push!(toks, id)                             # Terminate if EOS predicted, else append token and repeat
        end
        push!(names, join(chars[i] for i in toks[2:end]))                   # Decode integers back to text and save to output
    end
    names                                                                   
end

#   SCENE 2 : The Curtain Falls
println("\n INFERENCE (new, hallucinated words) :")                         
for (i, nm) in enumerate(infer(20)); println("sample $(lpad(i,2)): $nm"); end # Print the final generations
