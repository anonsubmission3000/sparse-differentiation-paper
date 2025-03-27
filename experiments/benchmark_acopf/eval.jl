using DrWatson
quickactivate(@__DIR__)

using DataFrames
using LaTeXStrings
using SummaryTables
using Typst_jll
using Printf: @sprintf

## Load results
df = collect_results(datadir("results"))

function extract_case_number(case_name::AbstractString)
    m = match(r"(\d+)", case_name)
    return parse(Int, m.captures[1])
end

# add normalized columns
df = transform(
    df,
    [:case_name] => ByRow(extract_case_number) => :case_number,
    # Sparsity detection
    [:time_sp_sym, :time_sp_sct] => ByRow(/) => :ratio_detection_sct,
    :time_sp_sym => ByRow(x -> 1) => :ratio_detection_sym,
    # AD
    [:time_dense_prepped, :time_sparse_unprepped] => ByRow(/) => :ratio_sparse_unprepped,
    [:time_dense_prepped, :time_sparse_prepped] => ByRow(/) => :ratio_sparse_prepped,
    :time_dense_prepped => ByRow(x -> 1) => :ratio_dense_prepped,
)
sort!(df, :case_number)

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

case_name = map(name -> name[15:end], df[:, :case_name]) # strip "pglib_opf_case_" prefix
inputs = df[:, :cols]
rel_sparsity = df[:, :rel_sparsity] .|> percentage_string
colors = df[:, :n_colors_col]
timing_sparsity_detection = Matrix(df[:, [:time_sp_sym, :time_sp_sct]])
ratios_detection = Matrix(df[:, [:ratio_detection_sym, :ratio_detection_sct]])
timings_ad = Matrix(
    df[:, [:time_dense_prepped, :time_sparse_prepped, :time_sparse_unprepped]]
)
ratios_ad = Matrix(
    df[:, [:ratio_dense_prepped, :ratio_sparse_prepped, :ratio_sparse_unprepped]]
)

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

walltime_str = "Wall time in seconds."

#===========#
# Detection #
#===========#

detection_ratio_str = "In parentheses: Wall time ratio compared to Symbolics.jl's sparsity detection (higher is better)."

header_detection = [
    "Problem",
    "Problem",
    "Sparsity",
    "Sparsity",
    Annotated("Sparsity detection", walltime_str),
    Annotated("Sparsity detection", walltime_str),
    Annotated("Sparsity detection", walltime_str),
]
vars_detection = [
    "Name",
    "Inputs",
    "Zeros",
    Annotated("Colors", "Number of colors resulting from greedy symmetric coloring."), # column colors
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
            Cell.(case_name, italic=true, halign=:left),
            Cell.(inputs),
            Cell.(rel_sparsity),
            Cell.(colors),
            cells_detection,
            cells_ratios_detection[:, 2],
        ),
    ),
)

short_table_range = floor.(Int, range(1, nrow(df); length=6))
table_detection_short = Table(
    vcat(
        Cell.(header_detection, bold=true, merge=true, border_bottom=true)',
        Cell.(vars_detection, merge=true, border_bottom=true)',
        hcat(
            Cell.(case_name[short_table_range], italic=true, halign=:left),
            Cell.(inputs[short_table_range]),
            Cell.(rel_sparsity[short_table_range]),
            Cell.(colors[short_table_range]),
            cells_detection[short_table_range, :],
            cells_ratios_detection[short_table_range, 2],
        ),
    ),
)

#==========#
# Autodiff #
#==========#

ratio_str = "In parentheses: Wall time ratio compared to prepared AD (higher is better)."

header = [
    "Problem",
    "Problem",
    "Sparsity",
    "Sparsity",
    # Annotated("Sparsity detection", walltime_str),
    Annotated("Hessian computation", walltime_str), # Dense prep
    Annotated("Hessian computation", walltime_str), # Sym prep
    Annotated("Hessian computation", walltime_str), # Sym prep (ratio)
    Annotated("Hessian computation", walltime_str), # Sym no prep
    Annotated("Hessian computation", walltime_str), # Sym no prep (ratio)
]
vars = [
    "Name",
    "Inputs",
    "Zeros",
    Annotated("Colors", "Number of colors resulting from greedy symmetric coloring."), # column colors
    # "SCT", # sparsity detection
    "AD (prepared)", # AD
    Annotated("ASD (prepared)", ratio_str),
    Annotated("ASD (prepared)", ratio_str),
    Annotated("ASD (unprepared)", ratio_str),
    Annotated("ASD (unprepared)", ratio_str),
]
@assert length(header) == length(vars)

table_ad = Table(
    vcat(
        Cell.(header, bold=true, merge=true, border_bottom=true)',
        Cell.(vars, merge=true, border_bottom=true)',
        hcat(
            Cell.(case_name, italic=true, halign=:left),
            Cell.(inputs,),
            Cell.(rel_sparsity),
            Cell.(colors),
            # cells_timing_detection,
            cells_timing_ad[:, 1],
            cells_timing_ad[:, 2],
            cells_ratios_ad[:, 2],
            cells_timing_ad[:, 3],
            cells_ratios_ad[:, 3],
        ),
    ),
)

table_ad_short = Table(
    vcat(
        Cell.(header, bold=true, merge=true, border_bottom=true)',
        Cell.(vars, merge=true, border_bottom=true)',
        hcat(
            Cell.(case_name[short_table_range], italic=true, halign=:left),
            Cell.(inputs[short_table_range]),
            Cell.(rel_sparsity[short_table_range]),
            Cell.(colors[short_table_range]),
            # cells_timing_detection,
            cells_timing_ad[short_table_range, 1],
            cells_timing_ad[short_table_range, 2],
            cells_ratios_ad[short_table_range, 2],
            cells_timing_ad[short_table_range, 3],
            cells_ratios_ad[short_table_range, 3],
        ),
    ),
)

function save(table, file_name)
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

save(table_detection, "opf_detection_benchmark_table")
save(table_detection_short, "opf_detection_benchmark_table_short")
save(table_ad, "opf_ad_benchmark_table")
save(table_ad_short, "opf_ad_benchmark_table_short")
