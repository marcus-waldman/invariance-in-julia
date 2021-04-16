clearconsole()

using Feather
using DataFrames
#import XLSX
#using RCall
#import Query
    #What is the difference between 'import' and 'using'?
    # because the filtering from Query doesn't work without the below code code
#using Query, DataFrames
#using DataFramesMeta
#using BenchmarkTools
    #Above used for @btime macro to time execution.
    # Note what a macro is
    # Note that it conflicts with Flux.params
#using FreqTables

dropbox_wd = string("D:/Dropbox")
analytic_wd = string(dropbox_wd, "/UNMC/pritzker-project/package/analytic-data/")
dashboard_wd = string(dropbox_wd, "/UNMC/pritzker-project/dashboard")
meta_wd = string(dashboard_wd,"/meta")
dif_wd = string(dropbox_wd,"/UNMC/pritzker-project/analysis/invariance")

cd(dif_wd)
long = Feather.read("long-gsednetbl-18Mar2021.feather")
first(long,5)
#long[:,"yij"] = convert(Vector{Int}, long[:,"yij"]) #This didn't actually convert it
#long[!, :yij] = convert(Int, long[!,:yij]) #This one throws an error.
long[!, :yij] = convert(Vector{Int}, long[!,:yij])
    # I have no idea what the hell the ! is doing here

refdf = DataFrame(j = 1:139, lex_ne = "")
for j = 1:139
    refdf[j,:lex_ne] = string("C", j)
end

size(long)
long = leftjoin(long,refdf,on = :lex_ne)


# Get a wide data set
Y = long[:,["mrwid","j","yij"]]
Y = unstack(Y, :mrwid, :j, :yij ) #converts from wide to long
Y = Y[:, filter(x -> x != "mrwid", names(Y))] #gets rid of the mrwid

#id = 2473
pairdf = DataFrame(mrwid = Int64[], yijlo = Int64[],
                   jlo = Int64[], jhi = Int64[],
                   Δy = Int64[], y10 = Int64[], y01 = Int64[]) #initializes the data frame. Note you need to specify variable types
for id in unique(long[:,:mrwid])

    # Slice to individual and sort by j
    #longi = long[filter(x -> x .== id, long[:,:mrwid]), :] #Note that this returns the WRONG SET OF ROWS (MRWID = 142. wtf??)
    longi = long[long[:,:mrwid].==id, :]
    sort!(longi, :j)

    Ji = nrow(longi)
    jveci = unique(longi[:,:j])

    jlo = 1
#    println(string("jlo=",jlo))
    tmp = DataFrame(mrwid = id, yijlo = longi[jlo,:yij],
                     jlo = jveci[jlo],
                     jhi = jveci[(jlo+1):Ji],
                     Δy = longi[(jlo+1):Ji,:yij].-longi[jlo,:yij],
                     y10 = 0, y01 = 0)
    tmp = tmp[tmp[:,:Δy] .!= 0,:]

    for jlo = 2:(Ji-1)
#        println(string("jlo=",jlo))
        tmp2 = DataFrame(mrwid = id, yijlo = longi[jlo,:yij],
                         jlo = jveci[jlo],
                         jhi = jveci[(jlo+1):Ji],
                         Δy = longi[(jlo+1):Ji,:yij].-longi[jlo,:yij],
                         y10 = 0, y01 = 0)
        tmp2 = tmp2[tmp2[:,:Δy] .!= 0,:]
        append!(tmp,tmp2)
    end

    append!(pairdf,tmp)
end
pairdf[:, :y10] = (pairdf[:,:Δy].==1)
pairdf[:, :y01] = (pairdf[:,:Δy].==-1)

x = pairdf[:,[:jlo,:jhi,:y01]]
x[!, :y01] = convert(Vector{Float64}, x[!, :y01])
rename!(x,["jlo","jhi","y"])

function m(ŷ::Vector{Float64}, df::DataFrame, 𝜹::Vector{Float64})
        # Inputs
        #   df - DataFrame with each row a pairwise comparisons.
        #           jlo (Int64) - indexes the lower-indexed item in the comparison,
        #           jhi (Int64) - indexes the higher-indexed item in the comparison,
        #           y (Float64) - indicates whether the lower-indexed item was passed,
        #                          while the higher-indexed item was not passed.
        #   𝜹 - Vector{Float64} of item difficulties.
        # Output: ŷ the predicted probability that
        # Extract the relevant input/response information from the DataFrame
        jlo = df[:,:jlo]; jhi = df[:,:jhi];

        # For model identification, the first parameter is constrained to take on a value of zero
        𝜹c = [0; 𝜹]

        #Complute fij (like yhat) function (see Zwinderman, 1995, Eqns 1-3)
        ŷ = exp.(𝜹c[jlo]) ./ (exp.(𝜹c[jlo]) .+ exp.(𝜹c[jhi]))

        return ŷ # \hat{y}
end

function fg(ŷ::Vector{Float64},df::DataFrame, J::Int64)
    y = df[:,:y]
    𝓁oss = -1.0*sum(y.*log.(ŷ) + y.*log.(1.0 .- ŷ))

    ∂𝜹c = zeros(J+1)
    for i in 1:nrow(df)
        jilo = df[i,:jlo]
        jihi = df[i,:jhi]

        ∂i = y[i]*(1.0-ŷ[i]) -  (1.0-y[i])*ŷ[i]

        ∂𝜹c[jilo] -= ∂i #flipped sign to make a minimization problem
        ∂𝜹c[jihi] += ∂i #flipped sign to make a minimization problem
    end

    return 𝓁oss, ∂𝜹c[2:J]
end


J = 138
𝜹 = zeros(J)
ŷ = zeros(nrow(pairdf))


ŷ = m(ŷ, x, 𝜹)
f, g = fg(ŷ,x,J)


ŷ
#function loss(yhat,y)
#    return -1.0*sum(y.*log.(yhat) + y.*log.(1.0 .- yhat))
#end


opt = ADAM()
max_grad(𝜽) = maximum(abs.(grads[𝜽]))

epochs = 100
for epoch = 1:epochs
  Flux.train!(loss, params(𝜹),  [(X,y)], opt)
  if (epoch % 10) == 0
      @show max_grad(𝜹)
      @show loss(X,y)
  end
end


grads = gradient( ()->loss(pairdf), params(𝜹))
grads[𝜹]

# I think what I woudl need to do would be to
# make the loss consistent with a model function m which would return yhat (i.e. fij)
# and then cross entropy.
# Ennding iwth this section. hopefully this jogs the memory when I return.
# "  data, labels = rand(10, 100), fill(0.5, 2, 100)
#   loss(x, y) = sum(Flux.crossentropy(m(x), y))
#   Flux.train!(loss, params(m), [(data,labels)], opt) "
# https://fluxml.ai/tutorials/2020/09/15/deep-learning-flux.html

## The below works, but I want to try to do this uing Flux

#using Optim
#using NLSolversBase
#using StatsPlots
#
# function objectivefn(params, df::DataFrame)
#
#     # Extract the relevant input/response information from the DataFrame
#     j_lo = df[:,:j_lo]
#     j_hi = df[:,:j_hi]
#     y10 = df[:,:y10]
#     y01 = df[:,:y01]
#
#     # For model identification, the first parameter is constrained to take on a value of zero
#     𝜹 = [0; params]
#
#     #Complute the pseudo-likelihood function (see Zwinderman, 1995, Eqns 1-3)
#     fij = exp.(𝜹[j_lo]) ./ (exp.(𝜹[j_lo]) .+ exp.(𝜹[j_hi]))
#     logl = sum(y10.*log.(fij) + y01.*log.(1.0 .- fij))
#
#     # Make a minimization problem and return the negative of the pseudolikelihood
#     return -logl
#
# end
#
#
# J = 138 # number of free parameters
# params_init = collect(range(0,1,length = J)) #initialize the parameter values
# objectivefn(params_init, pairdf)
# func = OnceDifferentiable(params -> objectivefn(params,pairdf), params_init; autodiff =:forward)
#
# #http://julianlsolvers.github.io/Optim.jl/v0.9.3/user/minimization/
#
# opt = optimize(func, params_init, AcceleratedGradientDescent(), Optim.Options(show_trace=true, iterations = 100))
# opt = optimize(func, Optim.minimizer(opt), LBFGS(), Optim.Options(show_trace=true, iterations = 100))
#
#
# using StatsPlots
# scatter(1:139,[0;Optim.minimizer(opt)])
