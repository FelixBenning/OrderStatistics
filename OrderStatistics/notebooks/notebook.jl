### A Pluto.jl notebook ###
# v0.12.7

using Markdown
using InteractiveUtils

# ╔═╡ 98aec1c2-21e1-11eb-31dd-a19c407f373c
begin
	using OrderStatistics:sample_extreme_values, diff_largest_second
	using Statistics:mean,std
	using Plots:plot, plot!, svg
	using LaTeXStrings
end

# ╔═╡ dca3e9a0-21e1-11eb-22b4-07c891533980
sample_extreme_values(10^6, 10^4, x->x, diff_largest_second)

# ╔═╡ 32c4fca2-2392-11eb-133e-213db77f04d1
times = [10, 10^2, 10^3, 5*10^3, 10^4, 5*10^4, 10^5, 10^6]

# ╔═╡ bcc952c0-2392-11eb-1e54-0b4d8fa5c73b
begin
	unif_mean = [
		mean(sample_extreme_values(t, 10^4, x->x, diff_largest_second))
		for t in times
	]
	exp_mean = [
		mean(sample_extreme_values(t, 10^4, x->-log(x), diff_largest_second))
		for t in times
	]
	exp_mean_2 = [
		mean(sample_extreme_values(t, 10^4, x->-log(x)/2, diff_largest_second))
		for t in times
	]
	exp_mean_05 = [
		mean(sample_extreme_values(t, 10^4, x->-log(x)*2, diff_largest_second))
		for t in times
	]
	heavy_tailed_mean = [
		mean(sample_extreme_values(t, 10^4, x->1/sqrt(x), diff_largest_second))
		for t in times
	]
end

# ╔═╡ d1d3f3c0-2395-11eb-00ea-59aa6de8c60f
begin
	p = plot(
		xaxis=("sample size",:log), 
		yaxis=("mean(largest - second_largest)",:log)
	)
	plot!(p, times, unif_mean, label ="U[0,1]", markershape=:auto)
	plot!(p, times, exp_mean, label = "Exp(1)", markershape=:auto)
	plot!(p, times, exp_mean_2, label = "Exp(2)", markershape=:auto)
	plot!(p, times, exp_mean_05, label = "Exp(1/2)", markershape=:auto)
	plot!(
		p, times, heavy_tailed_mean, 
		label = "$(L"F(x)=\left(1-\frac{1}{x^2}\right)\mathbb{1}_{x>1}")",
		markershape=:auto
	)
	p
end

# ╔═╡ 0246d1e0-239f-11eb-12d4-2d48509d4137
svg(p, "plot.svg")

# ╔═╡ Cell order:
# ╠═98aec1c2-21e1-11eb-31dd-a19c407f373c
# ╠═dca3e9a0-21e1-11eb-22b4-07c891533980
# ╠═32c4fca2-2392-11eb-133e-213db77f04d1
# ╠═bcc952c0-2392-11eb-1e54-0b4d8fa5c73b
# ╠═d1d3f3c0-2395-11eb-00ea-59aa6de8c60f
# ╠═0246d1e0-239f-11eb-12d4-2d48509d4137
