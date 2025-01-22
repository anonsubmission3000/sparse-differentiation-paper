using DrWatson # YES
quickactivate(@__DIR__)

using SparseConnectivityTracer
using SparseConnectivityTracer: GradientTracer, IndexSetGradientPattern
using Flux, Metalhead

# Load image from URL
using HTTP, FileIO, ImageIO
using DataAugmentation

# Save sparsity pattern as image
using SparseMatrixColorings
using SparseMatrixColorings: show_colors
using Colors, ColorSchemes

const GT = GradientTracer{IndexSetGradientPattern{Int,Set{Int}}}
const detector = TracerSparsityDetector(; gradient_tracer_type=GT)

function draw_conv_pattern(;
    img_size=16,  # image height and width
    conv_size=5,  # size of Conv filter
    c_in=3,       # input color channels
    c_out=1,      # output color channels
    bs=2,         # batch size
)
    # Define simple conv layer
    conv = Conv((conv_size, conv_size), c_in => c_out, relu)

    # Random input image in given size
    input = randn(img_size, img_size, c_in, bs)

    # Detect sparsity
    @info "Detecting sparsity pattern..."
    pattern = jacobian_sparsity(conv, input, detector)
    @info "Got pattern of size $(size(pattern))."

    ## Color and save pattern
    @info "Coloring sparsity pattern..."
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
    algo = GreedyColoringAlgorithm(; decompression=:direct)
    result = coloring(pattern, problem, algo)

    @info "Saving image of sparsity pattern..."
    ncolors = maximum(column_colors(result)) # for partition=:column
    colorscheme = get(ColorSchemes.rainbow, range(0.0, 1.0; length=ncolors))
    img = show_colors(result; colorscheme=colorscheme)
    path = plotsdir(
        "conv", "conv_$(conv_size)x$(conv_size)_$(c_in)_to_$(c_out)_bs$(bs).png"
    )
    save(path, img)
    @info "Saved color image to $path."

    colorscheme = [RGB(0, 0, 0)]
    img = show_colors(result; colorscheme=colorscheme, warn=false)
    path = plotsdir(
        "conv", "conv_$(conv_size)x$(conv_size)_$(c_in)_to_$(c_out)_bs$(bs)_black.png"
    )
    save(path, img)
    @info "Saved b&w image to $path."
end

for bs in (1, 2)
    for c_out in (1, 2)
        draw_conv_pattern(; bs, c_out)
    end
end
