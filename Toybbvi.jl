module Toybbvi
using Distributions
using Flux
using Flux: train!, @epochs
using Plots
using LinearAlgebra

# hyperparameters
const D = 2
const num_samples = 100
const steps = 200
const lr = 0.001 * num_samples
const make_gif = true

export fit
export plot_results
export reset_bbvi

# functions to do blackbox variational inference
function log_density(mu, log_sigma)
  # Calculate log density, hardcoded for 2d
  # input is either Float64 or TrackedReal{Float64}
  d1 = Normal(0., 1.35)
  d2 = Normal(0., exp(log_sigma))
  d1_density = logpdf(d1, log_sigma)
  d2_density = logpdf(d2, mu)
  d1_density + d2_density
end

log_density(p) = log_density(p...)  # accept tuples/arrays

gaussian_entropy(log_std) = .5 * D * (1. + log(2. + pi)) + sum(log_std)  # calc gaussian ent

function create_samples(num_samples; D=D, mu=mu, log_std=sigma)
  # data generating process, creates batch of size num_samples
  [[rand(Normal(), D, num_samples) .* exp.(log_std) .+ mu]]
end

function variational_objective(batch::TrackedArray{Float64}; log_std=sigma)
  # objective/loss function. input is batch of samples
  global curr_elbo
  log_px = mapslices(log_density, batch; dims=1)
  elbo = gaussian_entropy(log_std) + mean(log_px)
  curr_elbo = -elbo.data
  -elbo
end

function reset_bbvi()
  global mu, sigma, params, Z_q_traj, elbows
  mu = Tracker.param([-1, -1])
  sigma = Tracker.param([-5, -5])
  params = Tracker.Params([mu, sigma])
  Z_q_traj = Array{Array{Float32, ndims(X)}}(undef, steps)
  elbows = Array{Float32}(undef, steps)

  return nothing
end

# create initial parameters which will be optimized
mu = Tracker.param([-1, -1])
sigma = Tracker.param([-5, -5])
params = Tracker.Params([mu, sigma])
# temp var to pass loss from loss to callback
curr_elbo = undef

# Constants for plotting stuff
const x = 3.
const y = 5.
const x_rng = -x:0.1:x
const y_rng = -y:0.1:y
const X = repeat(reshape(x_rng, 1, :), length(y_rng), 1)
const Y = repeat(y_rng, 1, length(x_rng))
const Z = [exp(log_density([X[i], Y[i]])) for i in CartesianIndices(X)]

# create empty arrays to accumlate stuff during training for plotting
Z_q_traj = Array{Array{Float32, ndims(X)}}(undef, steps)
elbows = Array{Float32}(undef, steps)

# create optimizer
opt = ADAM(lr)

# callback function during training
function cbfn(step)
  q = MultivariateNormal(mu.data, Diagonal(exp.(2. * sigma.data)))
  Z_q_traj[step] = [pdf(q, [X[i], Y[i]]) for i in CartesianIndices(X)]
  elbows[step] = curr_elbo
  #println("loss: ", elbows[step])  # print loss
  return nothing
end

function fit()
  # training loop
  for step in 1:steps
    train!(variational_objective, params,
           create_samples(num_samples, D=2), opt,
           cb = () -> cbfn(step))
  end

  # print some lengths
  println("length of Z_q_traj: ", length(Z_q_traj))
  println("length of elbows: ", length(elbows))
end

# Pretty Pictures!
function plot_results()
  anim = @animate for step in 1:length(Z_q_traj)
    plot(contour(x_rng, y_rng, Z))
    contour!(x_rng, y_rng, Z_q_traj[step], color=:viridis)
  end

  gif(anim, "mygif.gif", fps=10)
end

end  # module
