
#table(
    rows: 8,
    columns: 9,
    column-gutter: 0.25em,
    align: (center, center, center, center, center, center, center, center, center),
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    [*Problem*],
    table.hline(y: 1, start: 0, end: 1, stroke: 0.75pt),
    table.cell(colspan: 5)[*Jacobian computation#super[1]*],
    table.hline(y: 1, start: 1, end: 6, stroke: 0.75pt),
    table.cell(colspan: 3)[*Newton step#super[1]*],
    table.hline(y: 1, start: 6, end: 9, stroke: 0.75pt),
    [N],
    table.hline(y: 2, start: 0, end: 1, stroke: 0.75pt),
    [\$\\makecell{\\text{AD} \\\\ \\text{(prepared)}}\$],
    table.hline(y: 2, start: 1, end: 2, stroke: 0.75pt),
    table.cell(colspan: 2)[\$\\makecell{\\text{ASD} \\\\ \\text{(prepared)}}\$#super[2]],
    table.hline(y: 2, start: 2, end: 4, stroke: 0.75pt),
    table.cell(colspan: 2)[\$\\makecell{\\text{ASD} \\\\ \\text{(unprepared)}}\$#super[2]],
    table.hline(y: 2, start: 4, end: 6, stroke: 0.75pt),
    [\$\\makecell{\\text{JVP} \\\\ \\text{(iterative)}}\$],
    table.hline(y: 2, start: 6, end: 7, stroke: 0.75pt),
    [\$\\makecell{\\text{Jacobian} \\\\ \\text{(iterative)}}\$],
    table.hline(y: 2, start: 7, end: 8, stroke: 0.75pt),
    [\$\\makecell{\\text{Jacobian} \\\\ \\text{(direct)}}\$],
    table.hline(y: 2, start: 8, end: 9, stroke: 0.75pt),
    [6],
    [\$1.64 \\cdot 10^{-5}\$],
    [\$\\mathbf{1.97 \\cdot 10^{-6}}\$],
    [*(8.3)*],
    [\$3.36 \\cdot 10^{-5}\$],
    [(0.5)],
    [\$\\mathbf{2.07 \\cdot 10^{-5}}\$],
    [\$2.19 \\cdot 10^{-5}\$],
    [\$4.52 \\cdot 10^{-5}\$],
    [12],
    [\$2.44 \\cdot 10^{-4}\$],
    [\$\\mathbf{8.67 \\cdot 10^{-6}}\$],
    [*(28.1)*],
    [\$1.70 \\cdot 10^{-4}\$],
    [(1.4)],
    [\$\\mathbf{1.34 \\cdot 10^{-4}}\$],
    [\$1.61 \\cdot 10^{-4}\$],
    [\$2.42 \\cdot 10^{-4}\$],
    [24],
    [\$4.02 \\cdot 10^{-3}\$],
    [\$\\mathbf{3.43 \\cdot 10^{-5}}\$],
    [*(117.2)*],
    [\$1.31 \\cdot 10^{-3}\$],
    [(3.1)],
    [\$\\mathbf{1.04 \\cdot 10^{-3}}\$],
    [\$1.34 \\cdot 10^{-3}\$],
    [\$1.24 \\cdot 10^{-3}\$],
    [48],
    [\$7.60 \\cdot 10^{-2}\$],
    [\$\\mathbf{1.68 \\cdot 10^{-4}}\$],
    [*(451.4)*],
    [\$1.70 \\cdot 10^{-2}\$],
    [(4.5)],
    [\$\\mathbf{8.34 \\cdot 10^{-3}}\$],
    [\$1.17 \\cdot 10^{-2}\$],
    [\$8.98 \\cdot 10^{-3}\$],
    [96],
    [\$1.35 \\cdot 10^{0}\$],
    [\$\\mathbf{6.68 \\cdot 10^{-4}}\$],
    [*(2017.2)*],
    [\$2.16 \\cdot 10^{-1}\$],
    [(6.2)],
    [\$7.56 \\cdot 10^{-2}\$],
    [\$1.11 \\cdot 10^{-1}\$],
    [\$\\mathbf{4.07 \\cdot 10^{-2}}\$],
    [192],
    [\$2.25 \\cdot 10^{1}\$],
    [\$\\mathbf{3.09 \\cdot 10^{-3}}\$],
    [*(7293.5)*],
    [\$4.62 \\cdot 10^{0}\$],
    [(4.9)],
    [\$1.05 \\cdot 10^{0}\$],
    [\$1.07 \\cdot 10^{0}\$],
    [\$\\mathbf{2.30 \\cdot 10^{-1}}\$],
    table.hline(y: 8, stroke: 1pt),
    table.cell(align: left, colspan: 9)[#text(size: 0.8em)[
        #super[1]Wall time in seconds.\
        #super[2]In parentheses: Wall time ratio compared to prepared AD (higher is better).
    ]],
)
