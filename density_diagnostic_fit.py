# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Fitting a power-law PDF to the density diagnostic trend
#
# This is an extension of the `density_diagnostic_trends` notebook, with the aim to
#
# 1. Perform a least squares fit of the model to the data
# 2. Calculate the credibility regions of the model parameters via MCMC
#
# Both can be handled by the `lmfit` library. 
#
# All the classes and functions from `density_diagnostic_trends.ipynb` have been extracted into the `ddtrends.py` module. 

import numpy as np
from astropy.table import Table
from matplotlib import pyplot as plt
import seaborn as sns
import corner
from lmfit import Model
import ddtrends as ddt

sns.set_context("notebook")

# ## Empirical density trend for Orion
#
# The following data was extracted from the Eduardo draft paper Figure 1, using ChatGPT

density_diagnostic_points = [
    # (diagnostic, log_nM, log_nobs, independent_subset)
    ("[S II] 6717/6731", 3.06, 2.309, True),
    ("[O II] 3727/3729", 3.16, 2.540, True),
    ("[Fe III] 4986/4881", 3.74, 2.685, True),
    ("[Fe III] 4658/4986", 3.84, 2.395, True),
    ("[Cl III] 5538/5518", 3.99, 3.070, True),
    ("[Fe III] 4986/4702", 4.04, 2.479, False),
    ("[S II] 4070/6717", 4.64, 2.593, False),
    ("[O II] 7320/3729", 4.69, 2.868, False),
    ("[S II] 4070/(6717+6731)", 4.85, 2.757, True),
    ("[S II] (4070+4075)/(6717+6731)", 4.91, 2.979, False),
    ("[O II] 7320/(3727+3729)", 5.03, 2.942, True),
    ("[O II] 7320/3727", 5.10, 3.127, False),
    ("[Fe III] 4702/4881", 6.19, 3.696, True),
    ("[Fe II] 8617/7155", 6.31, 3.333, True),
    ("[Fe III] 4658/4881", 6.36, 3.608, False),
    ("[Fe III] 4658/4702", 6.90, 3.457, False),
]

# Convert to an `astropy.Table` for ease of management

ori_tab = Table(
    rows=density_diagnostic_points,
    names=[
        "ratio",
        "log10(n_M)",
        "log10(n_obs)",
        "independent?",
    ],
)

# Display the full data table

ori_tab

# Display the subset of points with ratios that are algebraically independent

subset = ori_tab["independent?"]
ori_tab[subset]

# Plot the data with the linear trend fit and 95% confidence interval

ax = sns.scatterplot(ori_tab.to_pandas(), x="log10(n_M)", y="log10(n_obs)", hue="independent?")
xx = np.linspace(*ax.get_xlim())
ax.plot(xx, np.log10(ddt.n_obs_improved_fit(10**xx)), color="k", ls="dashed", zorder=0)
ax.fill_between(
    xx,
    np.log10(ddt.n_obs_improved_fit(10**xx, line_type="lower")),
    np.log10(ddt.n_obs_improved_fit(10**xx, line_type="upper")),
    color="0.8",
    zorder=-1,
)
sns.despine()

# ## Perform the fit to the data
#
# Unlike in the previous notebook, we will work exclusively in log-log space.
#
# First make convenient vectors of the data to be fitted.

xdata = ori_tab[subset]["log10(n_M)"].data
ydata = ori_tab[subset]["log10(n_obs)"].data
xdata, ydata


# Define a `lmfit.Model` object for the power-law PDF.
#
# Instead of including the upper density bound `log_n1` as a parameter, we instead use the width of the interval `log_width = log_n1 - log_n0`, which is the same as $\log_{10} n_\max / n_\min$. The reason is to allow us to enforce `log_n1 > log_n0`
#
# ***TODO*** *We still need to justify the variation of $\delta$ with $n_\mathcal{M}$*
#

# +
def plaw_model(x, m, log_n0, log_width):
    """
    Model function for log10_n_obs(log10_n_M) with parameters (m, log_n0, log_width)

    The upper bound on density is log_n1 = log_n0 + log_width

    We do not use log_n1 directly as a parameter since we want to make sure
    it is always greater than the lower bound. Also, it is very poorly constrained by the data

    Suitable for use with lmfit.Model
    """

    # Set up diagnostic line ratio parameters
    nM = 10 ** x
    # Vary with nM = 1e3 -> 1e7 as delta = 2 -> 200
    delta = 2.0 * np.sqrt(nM / 1e3)

    # Set up power-law PDF
    log_n1 = log_n0 + log_width
    pdf = ddt.PowerLawPDF(m, 10 ** log_n0, 10 ** log_n1)

    # Find observed density using the vectorized function
    n_obs = ddt.n_app_from_nM(nM, pdf, delta=delta)

    return np.log10(n_obs)
    

pmodel = Model(plaw_model)
# -

pmodel.param_names, pmodel.independent_vars

# Set initial values for parameters. We restrict the values, but over a broad range of:
# * $m = [-3.0, +0.5]$
# * $\log_{10} n_0 = [-3.0, +3.5]$
# * $\log_{10} n_1 / n_0 = [+0.1, +8.0]$
#
# This is only important for the upper limit, which otherwise tends to go ridiculously high. 

params = pmodel.make_params(
    m=dict(value=-0.5, min=-3, max=0.5),
    log_n0=dict(value=-1.0, min=-3, max=3.5),
    # log_width=dict(value=5.0, min=0.1, max=12),
    log_width=dict(value=5.0, min=0.1, max=8),
)
params

# Fit model to the data

sig_y = 0.25 * np.ones_like(ydata)

result = pmodel.fit(ydata, params, x=xdata, weights=1 / sig_y)

# +
from IPython.display import Markdown, display

def show_fit_md(result, heading="Fit result", level=4):
    hashes = "#" * level
    report = result.fit_report()
    display(Markdown(f"{hashes} {heading}\n\n```text\n{report}\n```"))


# -

show_fit_md(result, heading="LMfit result", level=3)

result.plot();

# So the best-fit values of power-law slope $m = -1.4 \pm 0.2$ and lower density bound $\log_{10} n_0 = 1.6 \pm 0.5$ are very similar to the values found by hand in the previous notebook.
#
# However, the upper density bound gets stuck at the highest allowed value $\log_{10} (n_1 / n_0) = 11.0$ with a nonsensical error estimate. *This gets fixed below in the MCMC fits*

# ## Find credibile bounds on PDF parameters with MCMC
#
# Use the `emcee` method, starting from the parameters of the previous fit.

mcmc_params = result.params.copy()
#mcmc_params.add("__lnsigma", value=np.log(0.1), min=np.log(0.001), max=np.log(1))

# The important thing here is to fix the estimated scatter of the data points a priori. This is set via the `weights=1 / sig_y` parameter to the fit. I use $\sigma_y = 0.25~\text{dex}$, which gives a reduced $\chi^2$ of about unity.
#
# If we do not do this, then the mcmc walkers spend all their time investigating the possibility that the scatter might be larger, which would mean the model would be more poorly constrained, and the resulting posteriors look a real mess.

emcee_kws = dict(
    steps=10_000,
    burn=1000,
    thin=1,
    seed=2026_06_09,
    progress=True,
    workers=1,
)

# I find 10,000 steps is required to avoid the autocorrelation warning. 
#
# I am leaving it single-threaded for now, since multiprocessing does not play nicely with a function defined in a notebook cell. I would have to move `plaw_model()` to a python file in order to use multiple workers. It takes a bit less than a minute to run on my laptop.

mcmc_result = pmodel.fit(
    data=ydata,
    x=xdata,
    weights=1 / sig_y,
    params=mcmc_params,
    method="emcee",
    fit_kws=emcee_kws,
) 

show_fit_md(mcmc_result, heading="MCMC result", level=3)

# ### Compare LMfit versus MCMC

result.params

mcmc_result.params

# The best value and uncertainty on $m$ are nearly identical. For the lower density bound, the value is the same, but the uncertainty is a bit higher with MCMC because it has found an extended tail towards low values (see below). 

fc = mcmc_result.flatchain
print("quantile", *mcmc_result.var_names)
for q in [0.05, 0.16, 0.50, 0.84, 0.95]:
    print(q, *[f"{fc[p].quantile(q):.2f}" for p in mcmc_result.var_names] )

posterior_samples = mcmc_result.flatchain.sample(n=100, random_state=1972_01_30)

# Pastel colored lines for the posterior sample
my_palette = sns.husl_palette(n_colors=101, s=0.9, l=0.75, h=0.8)
# Reserve first two colormap entries for data and best fit, with darker colors
my_palette[0] = (0, 0, 0)
my_palette[1] = (0.5, 0.2, 0)
with my_palette:
    g = mcmc_result.plot_fit()
    for d in posterior_samples.to_dict(orient="records"):
        g.plot(xx, mcmc_result.eval(x=xx, **d), alpha=1, lw=0.3, zorder=-1)
    # Replot the fit line over the full range, making sure to use the same color
    g.plot(xx, mcmc_result.eval(x=xx), lw=2, color=sns.color_palette()[1])

# ### Corner plot of MCMC result
#
# The first versions of this were looking really rough, but after a lot of work I have managed to get it to look pretty

red_cmap = sns.color_palette("light:salmon", n_colors=4)
red_cmap[0] = (1, 1, 1)
red_cmap

samples = mcmc_result.flatchain[["m", "log_n0", "log_width"]]
labels = [
    r"$m$",
    r"$\log_{10} (n_\min)$",
    r"$\log_{10}(n_\max/n_\min)$",
]
fig = corner.corner(
    samples,
    labels=labels,
    truths=[
        np.median(samples["m"]),
        np.median(samples["log_n0"]),
        np.median(samples["log_width"]),
    ],
    quantiles=[0.16, 0.50, 0.84],
    show_titles=True,
    title_fmt=".2f",
    title_kwargs={"fontsize": 12},
    label_kwargs={"fontsize": 15},
    bins=100,
    plot_datapoints=False,
    fill_contours=True,
    smooth=2,
    smooth1d=2,
    # Contours approximately enclosing 1, 2, 3 sigma for 2D Gaussian
    levels=(0.393, 0.865, 0.989),
    contour_kwargs={
        "linewidths": 1.2,
    },
    contourf_kwargs={
        # "cmap": "Reds",
        "colors": red_cmap,
    },
    # Axis/tick control
    max_n_ticks=4,
    top_ticks=False,
)
sns.despine()
fig.set_size_inches(7.0, 7.0)
fig.tight_layout()

# Contours show the 1, 2, and 3-sigma boundaries of the pairwise posterior distributions of model parameters. The blue point and lines show the median values. The dashed lines show the (16, 84)% quantiles ($\pm 1 \sigma$) of the marginal distributions. 

# ### Alternative corner plot that directly shows upper bound $\log_{10} n_\max$

samples["log_n1"] = samples["log_n0"] + samples["log_width"]
samples

labels = [
    r"$\beta$",
    r"$\log_{10} (n_\min)$",
    r"$\log_{10} (n_\max)$",
]
fig = corner.corner(
    samples[["m", "log_n0", "log_n1"]],
    labels=labels,
    truths=[
        np.median(samples["m"]),
        np.median(samples["log_n0"]),
        np.median(samples["log_n1"]),
    ],
    quantiles=[0.16, 0.50, 0.84],
    show_titles=True,
    title_fmt=".2f",
    title_kwargs={"fontsize": 12},
    label_kwargs={"fontsize": 12},
    bins=100,
    plot_datapoints=False,
    fill_contours=True,
    smooth=2,
    smooth1d=2,
    # Contours approximately enclosing 1, 2, 3 sigma for 2D Gaussian
    levels=(0.393, 0.865, 0.989),
    contour_kwargs={
        "linewidths": 1.2,
    },
    contourf_kwargs={
        # "cmap": "Reds",
        "colors": red_cmap,
    },
    # Axis/tick control
    max_n_ticks=4,
    top_ticks=False,
)
sns.despine()
fig.text(
    4/6, 
    5/6, 
    (
        "Uninformative prior:\n" 
        r"$\beta \sim [-2, 0.5]$" "\n"
        r"$\log_{10} (n_\min) \sim [-3, 3.5]$" "\n"
        r"$\log_{10} (n_\max / n_\min) \sim [0.1, 8]$" 
    ),
    ha="center",
    va="center",
    fontsize=14,
)
fig.text(0.08, 0.92, "a", ha="right", va="bottom", fontsize=20, fontweight="bold")
fig.set_size_inches(6.0, 6.0)
fig.tight_layout()
fig.savefig("plaw-density-distribution-mcmc-corner-uniform.pdf")

# This is better because it shows more clearly the two different components of the $m$ marginal distribution:
# 1. Major component (about 70% of the posterior) with the following characteristics:
#    * $m \approx -1.4 \pm 0.2$
#    * $\log_{10}(n_\min) \approx 1.6$
#    * $\log_{10}(n_\max) \ge 4$ but unconstrained upper limit to the upper bound.
# 2. Minor component (about 30% of the posterior) with the following characteristics:
#    * $m \approx -1.0$
#    * $\log_{10}(n_\min) \le 2.5$ but with unconstrained lower limit to the lower bound
#    * $\log_{10}(n_\max) \approx 4.5$
#
# We can try and separate out these two populations. See the section below. Thsi gives slightly different values for the best fits if we fix the slope at $-1.4$ and $-1.0$. We could do a better job by fitting a sum of two Gaussians to the marginal posterior $m$ distribution, but that hardly seems worth it.

# ## Calculate chacteristic densities for the fitted distributions
#
# The most important of these is the RMS density, since that is needed for the Strömgren condition.

pdf = ddt.PowerLawPDF(-1.3, 10**1.5, 10**6.3)
pdf.nrms()

# ### Add the RMS density to the posterior distribution graph
#
# Add a new column to `samples` dataframe with nrms

samples["log_nrms"] = samples.apply(
    lambda x: np.log10(
        ddt.PowerLawPDF(x["m"], 10 ** x["log_n0"], 10 ** x["log_n1"]).nrms()
    ),
    axis=1,
)
samples

labels = [
    r"$\beta$",
    r"$\log_{10} (n_\min)$",
    r"$\log_{10} (n_\max)$",
    r"$\log_{10} (n_\text{rms})$",
]
fig = corner.corner(
    samples[["m", "log_n0", "log_n1", "log_nrms"]],
    labels=labels,
    truths=[
        np.median(samples["m"]),
        np.median(samples["log_n0"]),
        np.median(samples["log_n1"]),
        np.median(samples["log_nrms"]),
    ],
    quantiles=[0.16, 0.50, 0.84],
    show_titles=True,
    title_fmt=".2f",
    title_kwargs={"fontsize": 12},
    label_kwargs={"fontsize": 15},
    bins=100,
    plot_datapoints=False,
    fill_contours=True,
    smooth=2,
    smooth1d=2,
    # Contours approximately enclosing 1, 2, 3 sigma for 2D Gaussian
    levels=(0.393, 0.865, 0.989),
    contour_kwargs={
        "linewidths": 1.2,
    },
    contourf_kwargs={
        # "cmap": "Reds",
        "colors": red_cmap,
    },
    # Axis/tick control
    max_n_ticks=4,
    top_ticks=False,
)
sns.despine()
fig.set_size_inches(7.0, 7.0)
fig.tight_layout()
# fig.savefig("plaw-density-distribution-mcmc-corner-uniform.pdf")

# ### Empirical constraints on $n_\text{rms}$ for Orion
#
# The Strömgren condition relates effective ionizing photon rate, i-front radius and rms density.
#
# Here is a table showing the variation in i-front radius, and hence RMS density for different regions of the nebula (the effective recomb coefficient and ionizing luminosity change a tiny bit, but this is relatively unimportant).
#
# | Representative region | $R_{\rm IF}$ (pc) | $Q_{\rm eff}$ (s$^{-1}$) | $T_e$ (K) | $\alpha_{B,e}$ (cm$^3$ s$^{-1}$) | $n_{\rm rms,e}$ (cm$^{-3}$) | $\log_{10} n_{\rm rms,e}$ | $f_\Omega$ | $f_A$ |
# |---|---:|---:|---:|---:|---:|---:|---:|---:|
# | Orion-S / closest optically thick fronts | 0.10 | $8.0\times10^{48}$ | 8500 | $2.74\times10^{-13}$ | $1.54\times10^{4}$ | 4.19 | 0.04 | 0.0004 |
# | Inner Huygens high-brightness MIF | 0.15 | $8.0\times10^{48}$ | 8400 | $2.76\times10^{-13}$ | $8.35\times10^{3}$ | 3.92 | 0.13 | 0.0032 |
# | Orion Bar / SE bright front | 0.20 | $8.5\times10^{48}$ | 8300 | $2.78\times10^{-13}$ | $5.57\times10^{3}$ | 3.75 | 0.08 | 0.0036 |
# | Outer Huygens / inner concave MIF | 0.30 | $8.5\times10^{48}$ | 8000 | $2.86\times10^{-13}$ | $2.99\times10^{3}$ | 3.48 | 0.18 | 0.0180 |
# | Inner Extended Orion Nebula | 0.50 | $9.0\times10^{48}$ | 7700 | $2.93\times10^{-13}$ | $1.41\times10^{3}$ | 3.15 | 0.13 | 0.0360 |
# | Extended nebula, intermediate scale | 1.00 | $9.5\times10^{48}$ | 7300 | $3.05\times10^{-13}$ | $5.04\times10^{2}$ | 2.70 | 0.11 | 0.1210 |
# | Outer extended nebula | 1.50 | $1.0\times10^{49}$ | 7200 | $3.08\times10^{-13}$ | $2.80\times10^{2}$ | 2.45 | 0.07 | 0.1730 |
# | SW extended halo / open blister wall | 2.00 | $1.0\times10^{49}$ | 7000 | $3.14\times10^{-13}$ | $1.80\times10^{2}$ | 2.26 | 0.05 | 0.2200 |
# | Far SW / maximum ionization-bounded scale | 3.00 | $1.0\times10^{49}$ | 7000 | $3.14\times10^{-13}$ | $9.8\times10^{1}$ | 1.99 | 0.03 | 0.2970 |
# | Matter-bounded / leaky directions | various| $1.0\times10^{49}$ | 7000 | $3.14\times10^{-13}$ | $<9.8\times10^{1}$ | $<1.99$ | 0.18 | 0.1280 |
#
# With the help of ChatGPT, I have estimated the solid-angle fractions, $f_\Omega$ and area fractions on the sky $f_A$ for each region. 
# * The $f_A$-weighted rms density distribution is appropriate for comparison with the Eduardo diagnostics study because he has used the median diagnostic ration over the face of the nebula. This is what we are approximating as $\log_{10} n_\text{rms} \sim 2.3 \pm 0.45$. 
# * The $f_\Omega$-weighted rms density distribution is relevant to the separation of the full 3D density fluctuations into angular and radial variations.
#
#
# The projected-area fractions are normalized with
# $$
# f_A \propto R_\text{IF}^2 f_\Omega
# $$
# for the ionization-bounded rows, but with the leaky/matter-bounded component assigned a separate projected-area fraction $f_A=0.128$ (pure guess, really). 
#
# Final prior:
# ```
# log_n_rms ~ Normal(mu=2.3, sigma=0.45)
# ```

# ### Look at correlation with the PDF parameters in more detail
#

joint = sns.jointplot(samples, x="log_n0", y="log_nrms", kind="hist", cmap="rocket_r", color="r")
xmin, xmax = joint.ax_joint.get_xlim()
ymin, ymax = joint.ax_joint.get_ylim()
xmin = min(xmin, ymin)
xmax = max(xmax, ymax)
joint.ax_joint.set_xlim(xmin, xmax)
joint.ax_joint.set_ylim(xmin, xmax)
joint.ax_joint.plot([xmin, xmax], [xmin, xmax], ls="dashed", color="k")

# So the RMS density is very well correlated with the minimum density
#
#
# Add the ratio $n_\text{rms} / n_\min$ to the `samples` dataframe

samples["log_nrms/n0"] = samples["log_nrms"] - samples["log_n0"]

# And look at correlation with all the parameters

# +
mask = (
    (samples["m"] > -2.0)
    & (samples["m"] < -0.5)
    & (samples["log_nrms/n0"] < 1.3)
)

sns.pairplot(
    samples[mask],
    kind="hist",
    x_vars=["m", "log_n0", "log_n1"],
    y_vars=["log_nrms", "log_nrms/n0"],
    plot_kws=dict(bins=100, pthresh=0.03, pmax=1, cmap="rocket_r"),
)
# -

# So, for $m < -1$ we have $n_\text{rms} \approx \text{2–3\ } n_\min $, increasing suddenly to $\approx 10\ n_\min $ for $m \approx -1$. 

# ## Using a physically motivated prior based on the Strömgren condition
#
# Since I had complained that the lognormal PDF might violate this condition, it would be hypocritical of me not to apply it here.
#
# The idea is to use the observed Strömgren radius, together with an estimate of the effective ionizing luminosity, $(1 - f_\text{esc})\, (1 - f_\text{d})\, Q_{49}$, to estimate the RMS density, $\langle n^2 \rangle^{1/2}$
#
# See "Empirical constraints ..." section above
#
# The prior we are going to use is $\log_{10} n_\text{rms} = 2.3 \pm 0.45$ with a Gaussian pdf:
# $$
# \log_{10} n_\text{rms} \sim \mathcal{N}(2.3, 0.45^2)
# $$
#
# Rather than redo the MCMC we will use the new prior as a weight for the posterior samples.  This should be fine since the support for the new prior is well within the range of the original one. 
#
# ***Disclaimer: the following cell was written in collaboration with ChatGPT***

# +
from scipy.stats import t, norm

# arrays from your existing flattened chain

m = samples["m"]
log_nmin = samples["log_n0"]
log_nmax = samples["log_n1"]
log_nrms = samples["log_nrms"]  # derived quantity already computed

# proposed prior on log10(n_rms)
mu = 2.3
sigma = 0.45
# PDF for a gaussian, ignoring constant subtractive terms since we normalize later
logw = -0.5 * ((log_nrms - mu) / sigma)**2

# stabilize and normalize
logw -= np.nanmax(logw)
w = np.exp(logw)
w /= np.sum(w)

# effective sample size
neff = 1.0 / np.sum(w**2)
print(f"N_eff = {neff:.0f} out of {len(w)}")


# -

# So the weighting has caused only a very modest reduction in the effective number of samples, which is good.

# +
def weighted_quantile(x, q, w):
    idx = np.argsort(x)
    xs = x[idx]
    ws = w[idx]
    cdf = np.cumsum(ws)
    return np.interp(q, cdf, xs)

for q in [0.0228, 0.1587, 0.5, 0.8413, 0.9772]:
    print(q, weighted_quantile(log_nrms, q, w))
# -

new_cmap = sns.color_palette("light:orange", n_colors=4)
new_cmap[0] = (1, 1, 1)
new_cmap

labels = [
    r"$\beta$",
    r"$\log_{10} (n_\min)$",
    r"$\log_{10} (n_\max)$",
]
fig = corner.corner(
    samples[["m", "log_n0", "log_n1"]],
    weights=w,
    labels=labels,
    truths=[
        weighted_quantile(samples["m"], 0.5, w),
        weighted_quantile(samples["log_n0"], 0.5, w),
        weighted_quantile(samples["log_n1"], 0.5, w),
    ],
    quantiles=[0.16, 0.50, 0.84],
    show_titles=True,
    title_fmt=".2f",
    title_kwargs={"fontsize": 12},
    label_kwargs={"fontsize": 12},
    bins=100,
    plot_datapoints=False,
    fill_contours=True,
    smooth=2,
    smooth1d=2,
    # Contours approximately enclosing 1, 2, 3 sigma for 2D Gaussian
    levels=(0.393, 0.865, 0.989),
    contour_kwargs={
        "linewidths": 1.2,
    },
    contourf_kwargs={
        # "cmap": "Reds",
        "colors": new_cmap,
    },
    # Axis/tick control
    max_n_ticks=4,
    top_ticks=False,
)
sns.despine()
fig.text(
    4/6, 
    5/6, 
    "Informative prior:\n" + r"$\log_{10} n_\text{rms} \sim \mathcal{N}(2.3, 0.45^2)$",
    ha="center",
    va="center",
    fontsize=14,
)
fig.text(0.08, 0.92, "b", ha="right", va="bottom", fontsize=20, fontweight="bold")
fig.set_size_inches(6.0, 6.0)
fig.tight_layout()
fig.savefig("plaw-density-distribution-mcmc-corner-informative-nrms.pdf")

# +
# fig.text?

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ## Find the posterior distribution of emission fractions above different density thresholds
#
# We want to find the fraction of the total emission from gas with density above a certain threshold (e.g., 1e6 pcc).  This is a determinate funtion of the power law parameters. Therefore, we can calculate it for each sample in the posterior `flatchain` samples, and then calculate the histogram.
# -

# ## Two-parameter fits with fixed $m = -1.4$ and $m = -1.0$
#
# Summary of results (see the "Compare results and corner plots" section)
# | Posterior component | $m$ | $\log_{10} n_\min$ | $\log_{10} n_\max$ | $\chi^2 / \nu$ |
# | :-- | :-- |:-- | :-- | :-- |
# | Major | $-1.4$  | $1.7 \pm 0.2$ | $ 7.4 \pm 1.5$ |  $0.90$|
# | Minor | $-1.0$  | $0.0 \pm 1.0$ | $ 4.7 \pm 0.4$ | $1.14$ |

params_major = result.params.copy()
params_major["m"].vary = False
params_major["m"].value = -1.4
params_major

result_major = pmodel.fit(ydata, params_major, x=xdata, weights=1 / sig_y)

params_minor = result.params.copy()
params_minor["m"].vary = False
params_minor["m"].value = -0.999
result_minor = pmodel.fit(ydata, params_minor, x=xdata, weights=1 / sig_y)

show_fit_md(result_major, heading="LMfit with fixed $m = -1.4$ ", level=3)

result_major.plot();

show_fit_md(result_minor, heading="LMfit with fixed $m = -1.0$", level=3)

result_minor.plot();



mcmc_result_major = pmodel.fit(
    data=ydata,
    x=xdata,
    weights=1 / sig_y,
    params=params_major,
    method="emcee",
    fit_kws=emcee_kws,
) 

show_fit_md(mcmc_result_major, heading="MCMC  with fixed $m = -1.4$ ", level=3)

mcmc_result_minor = pmodel.fit(
    data=ydata,
    x=xdata,
    weights=1 / sig_y,
    params=params_minor,
    method="emcee",
    fit_kws=emcee_kws,
) 

show_fit_md(mcmc_result_minor, heading="MCMC  with fixed $m = -1.0$ ", level=3)

# ### Compare results and corner plots

mcmc_result_major.params

mcmc_result_minor.params

samples_major = mcmc_result_major.flatchain[["log_n0", "log_width"]]
samples_major["log_n1"] = samples_major["log_n0"] + samples_major["log_width"]

labels = [
    r"$\log_{10} (n_\min)$",
    r"$\log_{10} (n_\max)$",
]
fig = corner.corner(
    samples_major[["log_n0", "log_n1"]],
    labels=labels,
    truths=[
        np.median(samples_major["log_n0"]),
        np.median(samples_major["log_n1"]),
    ],
    quantiles=[0.16, 0.50, 0.84],
    show_titles=True,
    title_fmt=".2f",
    title_kwargs={"fontsize": 12},
    label_kwargs={"fontsize": 15},
    bins=100,
    plot_datapoints=False,
    fill_contours=True,
    smooth=2,
    smooth1d=2,
    # Contours approximately enclosing 1, 2, 3 sigma for 2D Gaussian
    levels=(0.393, 0.865, 0.989),
    contour_kwargs={
        "linewidths": 1.2,
    },
    contourf_kwargs={
        # "cmap": "Reds",
        "colors": red_cmap,
    },
    # Axis/tick control
    max_n_ticks=4,
    top_ticks=False,
)
sns.despine()
fig.set_size_inches(7.0, 7.0)
fig.suptitle("Power law with fixed slope, $m = -1.4$")
fig.tight_layout()

samples_minor = mcmc_result_minor.flatchain[["log_n0", "log_width"]]
samples_minor["log_n1"] = samples_minor["log_n0"] + samples_minor["log_width"]

labels = [
    r"$\log_{10} (n_\min)$",
    r"$\log_{10} (n_\max)$",
]
fig = corner.corner(
    samples_minor[["log_n0", "log_n1"]],
    labels=labels,
    truths=[
        np.median(samples_minor["log_n0"]),
        np.median(samples_minor["log_n1"]),
    ],
    quantiles=[0.16, 0.50, 0.84],
    show_titles=True,
    title_fmt=".2f",
    title_kwargs={"fontsize": 12},
    label_kwargs={"fontsize": 15},
    bins=100,
    plot_datapoints=False,
    fill_contours=True,
    smooth=2,
    smooth1d=2,
    # Contours approximately enclosing 1, 2, 3 sigma for 2D Gaussian
    levels=(0.393, 0.865, 0.989),
    contour_kwargs={
        "linewidths": 1.2,
    },
    contourf_kwargs={
        # "cmap": "Reds",
        "colors": red_cmap,
    },
    # Axis/tick control
    max_n_ticks=4,
    top_ticks=False,
)
sns.despine()
fig.set_size_inches(7.0, 7.0)
fig.tight_layout()


