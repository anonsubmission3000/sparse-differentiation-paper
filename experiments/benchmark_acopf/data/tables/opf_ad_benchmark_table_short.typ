
#table(
    rows: 8,
    columns: 9,
    column-gutter: 0.25em,
    align: (left, center, center, center, center, center, center, center, center),
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    table.cell(colspan: 2, align: center)[*Problem*],
    table.hline(y: 1, start: 0, end: 2, stroke: 0.75pt),
    table.cell(colspan: 2)[*Sparsity*],
    table.hline(y: 1, start: 2, end: 4, stroke: 0.75pt),
    table.cell(colspan: 5)[*Hessian computation#super[1]*],
    table.hline(y: 1, start: 4, end: 9, stroke: 0.75pt),
    table.cell(align: center)[Name],
    table.hline(y: 2, start: 0, end: 1, stroke: 0.75pt),
    [Inputs],
    table.hline(y: 2, start: 1, end: 2, stroke: 0.75pt),
    [Zeros],
    table.hline(y: 2, start: 2, end: 3, stroke: 0.75pt),
    [Colors#super[2]],
    table.hline(y: 2, start: 3, end: 4, stroke: 0.75pt),
    [AD (prepared)],
    table.hline(y: 2, start: 4, end: 5, stroke: 0.75pt),
    table.cell(colspan: 2)[ASD (prepared)#super[3]],
    table.hline(y: 2, start: 5, end: 7, stroke: 0.75pt),
    table.cell(colspan: 2)[ASD (unprepared)#super[3]],
    table.hline(y: 2, start: 7, end: 9, stroke: 0.75pt),
    [_3\_lmbd_],
    [24],
    [91.15%],
    [6],
    [\$1.82 \\cdot 10^{-4}\$],
    [\$\\mathbf{8.29 \\cdot 10^{-5}}\$],
    [*(2.2)*],
    [\$1.45 \\cdot 10^{-4}\$],
    [(1.3)],
    [_60\_c_],
    [518],
    [99.56%],
    [12],
    [\$1.15 \\cdot 10^{-1}\$],
    [\$\\mathbf{2.36 \\cdot 10^{-3}}\$],
    [*(48.6)*],
    [\$8.61 \\cdot 10^{-3}\$],
    [(13.3)],
    [_240\_pserc_],
    [2558],
    [99.91%],
    [16],
    [\$3.51 \\cdot 10^{0}\$],
    [\$\\mathbf{2.50 \\cdot 10^{-2}}\$],
    [*(140.2)*],
    [\$1.04 \\cdot 10^{-1}\$],
    [(33.6)],
    [_1951\_rte_],
    [15018],
    [99.98%],
    [20],
    [\$2.00 \\cdot 10^{2}\$],
    [\$\\mathbf{1.54 \\cdot 10^{-1}}\$],
    [*(1293.4)*],
    [\$1.00 \\cdot 10^{0}\$],
    [(199.1)],
    [_2746wp\_k_],
    [19520],
    [99.99%],
    [14],
    [\$3.53 \\cdot 10^{2}\$],
    [\$\\mathbf{1.77 \\cdot 10^{-1}}\$],
    [*(1991.4)*],
    [\$1.51 \\cdot 10^{0}\$],
    [(234.5)],
    [_3375wp\_k_],
    [24350],
    [99.99%],
    [18],
    [\$6.25 \\cdot 10^{2}\$],
    [\$\\mathbf{2.54 \\cdot 10^{-1}}\$],
    [*(2463.9)*],
    [\$1.71 \\cdot 10^{0}\$],
    [(365.1)],
    table.hline(y: 8, stroke: 1pt),
    table.cell(colspan: 9)[#text(size: 0.8em)[
        #super[1]Wall time in seconds.\
        #super[2]Number of colors resulting from greedy column coloring.\
        #super[3]In parentheses: Wall time ratio compared to prepared AD (higher is better).
    ]],
)
