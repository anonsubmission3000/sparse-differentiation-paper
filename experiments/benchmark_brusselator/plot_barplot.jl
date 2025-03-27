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
filter!(r -> r.N <= 96, df)
sort!(df, :N)

## Draw figure
# use \the\textwidth in LaTeX document to print width in points
with_theme(theme_latexfonts()) do
    fig = Figure(; size=(465pt, 275pt), fontsize=9pt)

    for (i, r) in enumerate(eachrow(df))
        labels = ["ASD (SCT)", "ASD (Sym.)", "AD"] # plotted from bottom to top
        timings_pipeline_noprep = [
            r.time_asd_sct_noprep, r.time_asd_sym_noprep, r.time_ad_noprep
        ]
        timings_pipeline_prep = [r.time_asd_sct_prep, r.time_asd_sym_prep, r.time_ad_prep]
        timings_prep_with_detection = timings_pipeline_noprep .- timings_pipeline_prep
        timings_detection = [r.time_sp_sct, r.time_sp_sym, 0]

        ax = Axis(
            fig[i, 1];
            yticks=(1:3, labels),
            yticksvisible=false,
            ygridvisible=false,
            yticklabelsize=10,
        )
        barplot!(ax, timings_pipeline_noprep; color=cs[3], direction=:x)
        barplot!(ax, timings_prep_with_detection; color=cs[2], direction=:x)
        barplot!(ax, timings_detection; color=cs[1], direction=:x)

        xlims!(ax, 0, maximum(timings_pipeline_noprep))
        hidespines!(ax, :t, :r)

        Label(
            fig[i, 1, Right()],
            "N=$(r.N)";
            valign=:center,
            halign=:center,
            padding=(20, 0, 0, 0),
        )
    end

    # xlabel
    Label(
        fig[nrow(df), 1, Bottom()],
        "Walltime [s]";
        valign=:bottom,
        halign=:center,
        padding=(0, 0, 0, 20),
    )
    rowgap!(fig.layout, 6)

    # Legend
    group_color = [PolyElement(; color=c, strokecolor=:transparent) for c in cs[1:3]]
    Legend(
        fig[nrow(df) + 1, 1],
        group_color,
        ["Sparsity detection", "Memory allocation & coloring", "Jacobian-vector products"];
        patchsize=(13, 13),
        padding=(0, 0, 0, 0),
        orientation=:horizontal,
        framevisible=false,
    )

    ## Save figure
    save(plotsdir("brusselator_pipeline_small.png"), fig; px_per_unit=300 / inch)
    save(plotsdir("brusselator_pipeline_small.pdf"), fig)
    fig
end
