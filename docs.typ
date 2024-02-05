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

#let today = datetime.today()

#show: paper.with(
  title: [Semi- and Nonparametric Quantile Regression with Neural Networks],
  authors: (
    "Connacher Murphy",
  ),
  date: [#today.display("[month repr:long] [day], [year]")],
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

= Loss Functions
```
In progress.
```

== Standard Check Loss
The standard loss function used in quantile regression, often called the check loss, is defined
$
  cal(l)(u; tau)
  eq.def (tau - bb(1){u < 0}) u.
$<eqn:check-loss>
I plot this function in @fig:check-loss for various quantiles $tau$.

#figure(caption: "Check Loss Function")[
  #image(
    "out/loss_function_demonstration/check_loss.png",
    width: 80%
  )
]<fig:check-loss>
// We plot this function in Panel A of @fig:quantile-loss for various quantiles $tau$. Since the function is not differentiable at zero, we cannot apply a gradient descent approach. Even further, the second partial of the loss function is zero almost everywhere. We will not be able to invert $bold(Lambda)(bold(x))$ when we apply the #cite(<farrell_ea_2021>, form: "prose") framework.

== Smooth Check Loss
Next, I consider a smooth approximation to the check loss, proposed by #cite(<zheng_2011>, form: "prose"). I define
$
  cal(l)^(s)(u; tau, alpha) eq.def tau u + alpha log(1 + exp(-u / alpha)),
$<eqn:smooth-check-loss>
where we call $alpha$ the smoothing parameter. I plot the smooth check loss in @fig:smooth-check-loss for various quantiles $tau$ and smoothing parameters $alpha$.
#figure(caption: "Check Loss Function")[
  #stack(
    dir: ltr,
    image(
      "out/loss_function_demonstration/smooth_check_loss_tau_25.png",
      width: 50%
    ),
    image(
      "out/loss_function_demonstration/smooth_check_loss_tau_50.png",
      width: 50%
    )
  )
]<fig:smooth-check-loss>

== Simple Empirical Applications of (Smooth) Check Loss
I consider a simple empirical application of these two loss functions.

```
In progress.
```

#figure(caption: [Check Loss Function Performance on $X tilde U[0,1]$])[
  #stack(
    dir: ltr,
    image(
      "out/loss_function_demonstration/unconditional_quantiles_50_unif.png",
      width: 50%
    ),
    image(
      "out/loss_function_demonstration/unconditional_quantiles_75_unif.png",
      width: 50%
    )
  )
]<fig:unconditional-unif>

#figure(caption: [Check Loss Function Performance on $Y = X^2$, where $X tilde U[0,1]$])[
  #stack(
    dir: ltr,
    image(
      "out/loss_function_demonstration/unconditional_quantiles_50_unif_sq.png",
      width: 50%
    ),
    image(
      "out/loss_function_demonstration/unconditional_quantiles_75_unif_sq.png",
      width: 50%
    )
  )
]<fig:unconditional-unif>

#bibliography(("typst/zheng_2011.bib"), style:"chicago-author-date")

/*------------------------------------------------------------------------------
cp ~/projects/references/zheng_2011.bib ~/projects/quantile-dnn/typst/
------------------------------------------------------------------------------*/