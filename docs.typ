#import("typst/template_paper.typ"): *
#import("typst/common_commands.typ"): *

#set math.equation(numbering: "(1)")
#set figure.caption(position: top)

#show raw.where(block: true): block.with(
  fill: luma(250),
  stroke: luma(100) + 1pt,
  inset: 6pt,
  radius: 4pt,
)

#show quote: set align(center)

#show: paper.with(
  title: [Semi- and Nonparametric Quantile Regression with Neural Networks],
  authors: (
    "Connacher Murphy",
  ),
  date: "January 23, 2024",
)

#align(center)[
  ```
  This is a live document and subject to change.
  ```
]

#align(center)[
  *Abstract*
]

I consider the use of deep neural networks for quantile regression. The proposed framework allows for semi- and nonparametric estimation of the conditional quantile function.

```
In progress.
```
