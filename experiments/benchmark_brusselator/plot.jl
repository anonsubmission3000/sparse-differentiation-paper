using DrWatson # YES
quickactivate(@__DIR__)

using DataFrames
using LaTeXStrings
using CairoMakie

# These are relative to 1 CSS px
const inch = 96
const pt = 4 / 3
const cm = inch / 2.54

const cs = Makie.wong_colors()

## Load results
df = collect_results(datadir("results"))
sort!(df, :rel_sparsity)

N = df.N
t_ad_prep = df.time_ad_prep
t_asd_sct_prep = df.time_asd_sct_prep
t_asd_sct_noprep = df.time_asd_sct_noprep
t_asd_sym_prep = df.time_asd_sym_prep
t_asd_sym_noprep = df.time_asd_sym_noprep

## Draw figure
# use \the\textwidth in LaTeX document to print width in points
with_theme(theme_latexfonts()) do
    f = Figure(; size=(465pt, 140pt), fontsize=9pt)
    ax = Axis(
        f[1, 1]; xticks=N, xscale=log2, yscale=log10, xlabel=L"N", ylabel="Wall time [s]"
    )

    scatterlines!(ax, N, t_ad_prep; color=cs[1], marker=:rect, label="AD, prepared")
    scatterlines!(ax, N, t_asd_sct_prep; color=cs[3], label="ASD, prepared")
    scatterlines!(
        ax,
        N,
        t_asd_sct_noprep;
        color=cs[2],
        linestyle=:dash,
        label="ASD, unprepared using SCT",
    )
    scatterlines!(
        ax,
        N,
        t_asd_sym_noprep;
        color=cs[4],
        linestyle=:dash,
        label="ASD, unprepared using Symbolics",
    )

    f[1, 2] = Legend(f, ax; framevisible=false) #, orientation=:horizontal, nbanks=2)

    ## Save figure
    save(plotsdir("brusselator_benchmarks.png"), f; px_per_unit=300 / inch)
    save(plotsdir("brusselator_benchmarks.pdf"), f)
    f
end
