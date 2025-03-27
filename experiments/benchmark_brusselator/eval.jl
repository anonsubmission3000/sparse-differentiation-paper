using DrWatson # YES
quickactivate(@__DIR__)

using DataFrames
using LaTeXStrings
using SummaryTables
using Typst_jll
using Printf: @sprintf

## Load results
df = collect_results(datadir("results"))
# add normalized columns
df = transform(
    df,
    # Sparsity detection
    [:time_sp_sym, :time_sp_sct] => ByRow(/) => :ratio_detection_sct,
    :time_sp_sym => ByRow(x -> 1) => :ratio_detection_sym,
    # AD
    [:time_ad_prep, :time_asd_sct_noprep] => ByRow(/) => :ratio_asd_sct_noprep,
    [:time_ad_prep, :time_asd_sct_prep] => ByRow(/) => :ratio_asd_sct_prep,
    :time_ad_prep => ByRow(x -> 1) => :ratio_ad_prep,
)
sort!(df, :rel_sparsity)

# string formatting
two_digits(x) = @sprintf("%.2f", x)
print_ratio(x) = '(' * @sprintf("%.1f", x) * ')'
percentage_string(x) = @sprintf("%.2f", x * 100) * "%"

function two_digits_scientific(x, isbold=false)
    ex = floor(Int, log10(abs(x)))
    base = x / 10.0^ex
    base = round(base; digits=2)
    base_string = rpad(string(base), 4, '0') # ensure padding zeros for two digits
    ex_string = if signbit(ex)
        "-$(abs(ex))"
    else
        string(ex)
    end
    if isbold
        return L"\mathbf{%$base_string \cdot 10^{%$ex_string}}"
    else
        return L"%$base_string \cdot 10^{%$ex_string}"
    end
end
problems = Matrix(df[:, [:N, :cols, :rows]])
rel_sparsity = df[:, :rel_sparsity] .|> percentage_string
colors = df[:, :n_colors_col]
timing_sparsity_detection = Matrix(df[:, [:time_sp_sym, :time_sp_sct]])
ratios_detection = Matrix(df[:, [:ratio_detection_sym, :ratio_detection_sct]])
timings_ad = Matrix(df[:, [:time_ad_prep, :time_asd_sct_prep, :time_asd_sct_noprep]])
ratios_ad = Matrix(df[:, [:ratio_ad_prep, :ratio_asd_sct_prep, :ratio_asd_sct_noprep]])
timings_solve = Matrix(df[:, [:time_ls_op_iter, :time_ls_sp_iter, :time_ls_sp_direct]])

function cells_bold_smallest_scientific(A; halign=:center)
    bolds = fill(false, size(A)...)
    bolds[argmin(A; dims=2)] .= true
    return broadcast((x, b) -> Cell(two_digits_scientific(x, b); halign=halign), A, bolds)
end
function cells_bold_largest_along_row(A, tfm=identity; halign=:center)
    bolds = fill(false, size(A)...)
    bolds[argmax(A; dims=2)] .= true
    return broadcast((x, b) -> Cell(tfm(x); bold=b, halign=halign), A, bolds)
end

cells_detection = cells_bold_smallest_scientific(timing_sparsity_detection)
cells_ratios_detection = cells_bold_largest_along_row(ratios_detection, print_ratio)
cells_timing_ad = cells_bold_smallest_scientific(timings_ad)
cells_ratios_ad = cells_bold_largest_along_row(ratios_ad, print_ratio; halign=:center)
cells_timing_solve = cells_bold_smallest_scientific(timings_solve)

walltime_str = "Wall time in seconds."

#===========#
# Detection #
#===========#

detection_ratio_str = "In parentheses: Wall time ratio compared to Symbolics.jl's sparsity detection (higher is better)."

header_detection = [
    "Problem",
    "Problem",
    "Problem",
    "Sparsity",
    "Sparsity",
    Annotated("Sparsity detection", walltime_str),
    Annotated("Sparsity detection", walltime_str),
    Annotated("Sparsity detection", walltime_str),
]
vars_detection = [
    "N",
    "Inputs",
    "Outputs",
    "Zeros",
    Annotated("Colors", "Number of colors resulting from greedy column coloring."), # column colors
    "Symbolics", # sparsity detection
    Annotated("SCT", detection_ratio_str),
    Annotated("SCT", detection_ratio_str),
]
@assert length(header_detection) == length(vars_detection)

table_detection = Table(
    vcat(
        Cell.(header_detection, bold=true, merge=true, border_bottom=true)',
        Cell.(vars_detection, merge=true, border_bottom=true)',
        hcat(
            Cell.(problems),
            Cell.(rel_sparsity),
            Cell.(colors),
            cells_detection,
            cells_ratios_detection[:, 2],
        ),
    ),
)

#==========#
# Autodiff #
#==========#

ad_ratio_str = "In parentheses: Wall time ratio compared to prepared AD (higher is better)."

header_ad = [
    "Problem",
    Annotated("Jacobian computation", walltime_str), # Dense
    Annotated("Jacobian computation", walltime_str), # SCT
    Annotated("Jacobian computation", walltime_str), # SCT (ratio)
    Annotated("Jacobian computation", walltime_str), # Sym 
    Annotated("Jacobian computation", walltime_str), # Sym (ratio)
    Annotated("Newton step", walltime_str),
    Annotated("Newton step", walltime_str),
    Annotated("Newton step", walltime_str),
]
vars_ad = [
    "N",
    L"\makecell{\text{AD} \\ \text{(prepared)}}", # AD
    Annotated(L"\makecell{\text{ASD} \\ \text{(prepared)}}", ad_ratio_str),
    Annotated(L"\makecell{\text{ASD} \\ \text{(prepared)}}", ad_ratio_str),
    Annotated(L"\makecell{\text{ASD} \\ \text{(unprepared)}}", ad_ratio_str),
    Annotated(L"\makecell{\text{ASD} \\ \text{(unprepared)}}", ad_ratio_str),
    L"\makecell{\text{JVP} \\ \text{(iterative)}}", # AD
    L"\makecell{\text{Jacobian} \\ \text{(iterative)}}", # AD
    L"\makecell{\text{Jacobian} \\ \text{(direct)}}", # AD
]
@assert length(header_ad) == length(vars_ad)

table_ad = Table(
    vcat(
        Cell.(header_ad, bold=true, merge=true, border_bottom=true)',
        Cell.(vars_ad, merge=true, border_bottom=true)',
        hcat(
            Cell.(problems[:, 1]),
            cells_timing_ad[:, 1],
            cells_timing_ad[:, 2],
            cells_ratios_ad[:, 2],
            cells_timing_ad[:, 3],
            cells_ratios_ad[:, 3],
            cells_timing_solve[:, 1],
            cells_timing_solve[:, 2],
            cells_timing_solve[:, 3],
        ),
    ),
)

function save_table(table, file_name)
    ## LaTeX
    texfile = joinpath(datadir("tables"), "$file_name.tex")
    open(texfile, "w") do io
        # print the table as LaTeX code
        show(io, MIME"text/latex"(), table)
    end
    # Not rendering LaTeX table.

    ## Typst
    typfile = joinpath(datadir("tables"), "$file_name.typ")
    open(typfile, "w") do io
        # print the table as latex code
        show(io, MIME"text/typst"(), table)
    end
    # render the tex file to pdf
    Typst_jll.typst() do bin
        run(`$bin compile $typfile`)
    end
end

save_table(table_detection, "brusselator_detection_benchmark_table")
save_table(table_ad, "brusselator_ad_benchmark_table")
