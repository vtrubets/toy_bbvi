include("./Toybbvi.jl")
using .Toybbvi

@time fit()
#@time plot_results()
@time reset_bbvi()
@time fit()
#@time plot_results()
