
#table(
    rows: 43,
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
    [_5\_pjm_],
    [44],
    [94.99%],
    [8],
    [\$6.33 \\cdot 10^{-4}\$],
    [\$\\mathbf{1.71 \\cdot 10^{-4}}\$],
    [*(3.7)*],
    [\$3.03 \\cdot 10^{-4}\$],
    [(2.1)],
    [_14\_ieee_],
    [118],
    [97.84%],
    [10],
    [\$5.38 \\cdot 10^{-3}\$],
    [\$\\mathbf{4.84 \\cdot 10^{-4}}\$],
    [*(11.1)*],
    [\$1.12 \\cdot 10^{-3}\$],
    [(4.8)],
    [_24\_ieee\_rts_],
    [266],
    [99.22%],
    [12],
    [\$2.56 \\cdot 10^{-2}\$],
    [\$\\mathbf{1.04 \\cdot 10^{-3}}\$],
    [*(24.7)*],
    [\$2.74 \\cdot 10^{-3}\$],
    [(9.3)],
    [_30\_as_],
    [236],
    [98.89%],
    [12],
    [\$2.39 \\cdot 10^{-2}\$],
    [\$\\mathbf{1.10 \\cdot 10^{-3}}\$],
    [*(21.8)*],
    [\$2.84 \\cdot 10^{-3}\$],
    [(8.4)],
    [_30\_ieee_],
    [236],
    [98.89%],
    [12],
    [\$2.37 \\cdot 10^{-2}\$],
    [\$\\mathbf{1.09 \\cdot 10^{-3}}\$],
    [*(21.6)*],
    [\$2.87 \\cdot 10^{-3}\$],
    [(8.3)],
    [_39\_epri_],
    [282],
    [99.10%],
    [10],
    [\$3.28 \\cdot 10^{-2}\$],
    [\$\\mathbf{1.21 \\cdot 10^{-3}}\$],
    [*(27.1)*],
    [\$3.43 \\cdot 10^{-3}\$],
    [(9.6)],
    [_57\_ieee_],
    [448],
    [99.41%],
    [14],
    [\$8.80 \\cdot 10^{-2}\$],
    [\$\\mathbf{3.96 \\cdot 10^{-3}}\$],
    [*(22.2)*],
    [\$9.23 \\cdot 10^{-3}\$],
    [(9.5)],
    [_60\_c_],
    [518],
    [99.56%],
    [12],
    [\$1.15 \\cdot 10^{-1}\$],
    [\$\\mathbf{2.36 \\cdot 10^{-3}}\$],
    [*(48.6)*],
    [\$8.61 \\cdot 10^{-3}\$],
    [(13.3)],
    [_73\_ieee\_rts_],
    [824],
    [99.74%],
    [12],
    [\$2.75 \\cdot 10^{-1}\$],
    [\$\\mathbf{3.47 \\cdot 10^{-3}}\$],
    [*(79.1)*],
    [\$1.54 \\cdot 10^{-2}\$],
    [(17.8)],
    [_89\_pegase_],
    [1042],
    [99.74%],
    [26],
    [\$5.61 \\cdot 10^{-1}\$],
    [\$\\mathbf{1.61 \\cdot 10^{-2}}\$],
    [*(34.8)*],
    [\$4.28 \\cdot 10^{-2}\$],
    [(13.1)],
    [_118\_ieee_],
    [1088],
    [99.77%],
    [12],
    [\$5.55 \\cdot 10^{-1}\$],
    [\$\\mathbf{5.25 \\cdot 10^{-3}}\$],
    [*(105.8)*],
    [\$3.13 \\cdot 10^{-2}\$],
    [(17.7)],
    [_162\_ieee\_dtc_],
    [1484],
    [99.82%],
    [16],
    [\$1.16 \\cdot 10^{0}\$],
    [\$\\mathbf{1.53 \\cdot 10^{-2}}\$],
    [*(75.7)*],
    [\$5.53 \\cdot 10^{-2}\$],
    [(20.9)],
    [_179\_goc_],
    [1468],
    [99.83%],
    [14],
    [\$1.08 \\cdot 10^{0}\$],
    [\$\\mathbf{1.33 \\cdot 10^{-2}}\$],
    [*(81.3)*],
    [\$5.06 \\cdot 10^{-2}\$],
    [(21.4)],
    [_197\_snem_],
    [1608],
    [99.85%],
    [14],
    [\$1.34 \\cdot 10^{0}\$],
    [\$\\mathbf{1.46 \\cdot 10^{-2}}\$],
    [*(92.2)*],
    [\$5.84 \\cdot 10^{-2}\$],
    [(23.0)],
    [_200\_activ_],
    [1456],
    [99.82%],
    [12],
    [\$1.02 \\cdot 10^{0}\$],
    [\$\\mathbf{6.94 \\cdot 10^{-3}}\$],
    [*(146.6)*],
    [\$3.88 \\cdot 10^{-2}\$],
    [(26.3)],
    [_240\_pserc_],
    [2558],
    [99.91%],
    [16],
    [\$3.51 \\cdot 10^{0}\$],
    [\$\\mathbf{2.50 \\cdot 10^{-2}}\$],
    [*(140.2)*],
    [\$1.04 \\cdot 10^{-1}\$],
    [(33.6)],
    [_300\_ieee_],
    [2382],
    [99.89%],
    [14],
    [\$3.00 \\cdot 10^{0}\$],
    [\$\\mathbf{2.14 \\cdot 10^{-2}}\$],
    [*(140.3)*],
    [\$9.67 \\cdot 10^{-2}\$],
    [(31.1)],
    [_500\_goc_],
    [4254],
    [99.94%],
    [14],
    [\$1.18 \\cdot 10^{1}\$],
    [\$\\mathbf{3.85 \\cdot 10^{-2}}\$],
    [*(307.3)*],
    [\$2.20 \\cdot 10^{-1}\$],
    [(53.7)],
    [_588\_sdet_],
    [4110],
    [99.94%],
    [14],
    [\$1.14 \\cdot 10^{1}\$],
    [\$\\mathbf{3.60 \\cdot 10^{-2}}\$],
    [*(316.1)*],
    [\$2.14 \\cdot 10^{-1}\$],
    [(53.3)],
    [_793\_goc_],
    [5432],
    [99.95%],
    [14],
    [\$2.17 \\cdot 10^{1}\$],
    [\$\\mathbf{4.91 \\cdot 10^{-2}}\$],
    [*(443.1)*],
    [\$3.33 \\cdot 10^{-1}\$],
    [(65.3)],
    [_1354\_pegase_],
    [11192],
    [99.98%],
    [18],
    [\$1.36 \\cdot 10^{2}\$],
    [\$\\mathbf{1.21 \\cdot 10^{-1}}\$],
    [*(1128.4)*],
    [\$6.21 \\cdot 10^{-1}\$],
    [(219.6)],
    [_1803\_snem_],
    [15246],
    [99.98%],
    [16],
    [\$2.09 \\cdot 10^{2}\$],
    [\$\\mathbf{1.66 \\cdot 10^{-1}}\$],
    [*(1259.5)*],
    [\$1.07 \\cdot 10^{0}\$],
    [(195.0)],
    [_1888\_rte_],
    [14480],
    [99.98%],
    [18],
    [\$8.15 \\cdot 10^{2}\$],
    [\$\\mathbf{1.43 \\cdot 10^{-1}}\$],
    [*(5706.7)*],
    [\$8.76 \\cdot 10^{-1}\$],
    [(930.4)],
    [_1951\_rte_],
    [15018],
    [99.98%],
    [20],
    [\$2.00 \\cdot 10^{2}\$],
    [\$\\mathbf{1.54 \\cdot 10^{-1}}\$],
    [*(1293.4)*],
    [\$1.00 \\cdot 10^{0}\$],
    [(199.1)],
    [_2000\_goc_],
    [19008],
    [99.99%],
    [18],
    [\$3.58 \\cdot 10^{2}\$],
    [\$\\mathbf{2.15 \\cdot 10^{-1}}\$],
    [*(1669.5)*],
    [\$1.61 \\cdot 10^{0}\$],
    [(222.7)],
    [_2312\_goc_],
    [17128],
    [99.98%],
    [16],
    [\$2.75 \\cdot 10^{2}\$],
    [\$\\mathbf{1.87 \\cdot 10^{-1}}\$],
    [*(1470.7)*],
    [\$1.35 \\cdot 10^{0}\$],
    [(204.5)],
    [_2383wp\_k_],
    [17004],
    [99.98%],
    [16],
    [\$2.65 \\cdot 10^{2}\$],
    [\$\\mathbf{1.80 \\cdot 10^{-1}}\$],
    [*(1468.2)*],
    [\$1.14 \\cdot 10^{0}\$],
    [(231.4)],
    [_2736sp\_k_],
    [19088],
    [99.99%],
    [14],
    [\$3.30 \\cdot 10^{2}\$],
    [\$\\mathbf{1.78 \\cdot 10^{-1}}\$],
    [*(1857.2)*],
    [\$1.40 \\cdot 10^{0}\$],
    [(235.5)],
    [_2737sop\_k_],
    [18988],
    [99.99%],
    [16],
    [\$3.29 \\cdot 10^{2}\$],
    [\$\\mathbf{2.02 \\cdot 10^{-1}}\$],
    [*(1629.8)*],
    [\$1.47 \\cdot 10^{0}\$],
    [(223.0)],
    [_2742\_goc_],
    [24540],
    [99.99%],
    [14],
    [\$6.50 \\cdot 10^{2}\$],
    [\$\\mathbf{2.41 \\cdot 10^{-1}}\$],
    [*(2694.1)*],
    [\$1.78 \\cdot 10^{0}\$],
    [(366.3)],
    [_2746wop\_k_],
    [19582],
    [99.99%],
    [16],
    [\$3.64 \\cdot 10^{2}\$],
    [\$\\mathbf{2.07 \\cdot 10^{-1}}\$],
    [*(1755.7)*],
    [\$1.54 \\cdot 10^{0}\$],
    [(235.6)],
    [_2746wp\_k_],
    [19520],
    [99.99%],
    [14],
    [\$3.53 \\cdot 10^{2}\$],
    [\$\\mathbf{1.77 \\cdot 10^{-1}}\$],
    [*(1991.4)*],
    [\$1.51 \\cdot 10^{0}\$],
    [(234.5)],
    [_2848\_rte_],
    [21822],
    [99.99%],
    [20],
    [\$4.67 \\cdot 10^{2}\$],
    [\$\\mathbf{2.24 \\cdot 10^{-1}}\$],
    [*(2083.5)*],
    [\$1.80 \\cdot 10^{0}\$],
    [(259.7)],
    [_2853\_sdet_],
    [23028],
    [99.99%],
    [26],
    [\$5.38 \\cdot 10^{2}\$],
    [\$\\mathbf{3.62 \\cdot 10^{-1}}\$],
    [*(1486.9)*],
    [\$1.68 \\cdot 10^{0}\$],
    [(320.6)],
    [_2868\_rte_],
    [22090],
    [99.99%],
    [20],
    [\$5.02 \\cdot 10^{2}\$],
    [\$\\mathbf{2.35 \\cdot 10^{-1}}\$],
    [*(2137.9)*],
    [\$1.73 \\cdot 10^{0}\$],
    [(290.0)],
    [_2869\_pegase_],
    [25086],
    [99.99%],
    [28],
    [\$5.08 \\cdot 10^{2}\$],
    [\$\\mathbf{4.07 \\cdot 10^{-1}}\$],
    [*(1249.0)*],
    [\$1.99 \\cdot 10^{0}\$],
    [(255.5)],
    [_3012wp\_k_],
    [21082],
    [99.99%],
    [14],
    [\$4.33 \\cdot 10^{2}\$],
    [\$\\mathbf{1.96 \\cdot 10^{-1}}\$],
    [*(2208.3)*],
    [\$1.77 \\cdot 10^{0}\$],
    [(245.1)],
    [_3022\_goc_],
    [23238],
    [99.99%],
    [18],
    [\$5.76 \\cdot 10^{2}\$],
    [\$\\mathbf{2.51 \\cdot 10^{-1}}\$],
    [*(2296.9)*],
    [\$1.48 \\cdot 10^{0}\$],
    [(390.7)],
    [_3120sp\_k_],
    [21608],
    [99.99%],
    [18],
    [\$4.56 \\cdot 10^{2}\$],
    [\$\\mathbf{2.26 \\cdot 10^{-1}}\$],
    [*(2019.2)*],
    [\$1.90 \\cdot 10^{0}\$],
    [(240.1)],
    [_3375wp\_k_],
    [24350],
    [99.99%],
    [18],
    [\$6.25 \\cdot 10^{2}\$],
    [\$\\mathbf{2.54 \\cdot 10^{-1}}\$],
    [*(2463.9)*],
    [\$1.71 \\cdot 10^{0}\$],
    [(365.1)],
    table.hline(y: 43, stroke: 1pt),
    table.cell(colspan: 9)[#text(size: 0.8em)[
        #super[1]Wall time in seconds.\
        #super[2]Number of colors resulting from greedy column coloring.\
        #super[3]In parentheses: Wall time ratio compared to prepared prepared AD (higher is better).
    ]],
)
