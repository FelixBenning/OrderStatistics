### A Pluto.jl notebook ###
# v0.12.7

using Markdown
using InteractiveUtils

# ╔═╡ 98aec1c2-21e1-11eb-31dd-a19c407f373c
using OrderStatistics:sample_extreme_values, diff_largest_second

# ╔═╡ dca3e9a0-21e1-11eb-22b4-07c891533980
sample_extreme_values(10^6, 10^6, x->x; callable=diff_largest_second)

# ╔═╡ Cell order:
# ╠═98aec1c2-21e1-11eb-31dd-a19c407f373c
# ╠═dca3e9a0-21e1-11eb-22b4-07c891533980
