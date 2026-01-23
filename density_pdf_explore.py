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

from single_globule import Flow

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context("talk")

# ## Spherical globule
#
# Increase the number of points to get better histograms

Flow.Umax = 4.0
Flow.NR = 10_000
flow = Flow()

# ### Look at the velocity law

fig, ax = plt.subplots()
ax.plot(flow.R_edges, flow.U_edges)
ax.set_xlabel("Radius")
ax.set_ylabel("Mach number")
ax.set_ylim(0, None)

# ### Look at the density profile

fig, ax = plt.subplots()
ax.plot(flow.R_edges, flow.density_edges, label="Density: $n(R)$")
ax.plot(flow.R_edges, flow.R_edges**2 * flow.density_edges**2, label="Weight factor: $R^2 n^2$")
ax.set_xlabel("Radius")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()

# ## Calculating $H_n$ from the gradient of the velocity law
#
# ### General velocity law
# In general case (for spherical geometry) we have
# $$
# H_n \, dn = e \, dV = n^2 R^2 \, d R
# $$
# so that 
# $$
# H_n = \frac{n^2 R^2}{|d n / d R|}
# $$
# where
# $$
# \frac{d n}{d R} = \frac{d}{d R} U^{-1} R^{-2}
# = - 2 U^{-1} R^{-3} - R^{-2} U^{-2} \frac{d U}{d R}
# = \frac{- 2}{U R^3} \left( 1 + \frac{R}{U}\frac{d U}{d R}\right)
# $$
# Given that 
# $$
# n = \frac{1}{U R^2}
# $$
# we therefore have that 
# $$
# H_n = \frac{R^2 U R^3}{2 U^2 R^4 (1 + d \ln U / d \ln R)} = \frac{R}{2 U (1 + d \ln U / d \ln R)}
# $$

# ### Constant velocity wind
#
# In this case $U = 1$, $d\ln U / d\ln R = 0$, we get 
# $$
# H_n = \frac12 R = \frac12 n^{-1/2}
# $$
# which is exactly the same as I derived in the handwritten notes

# ### Dyson velocity law
#
# We can probably calculate the derivative analytically, but first I will do it by finite differences

dlnU_dlnR = np.gradient( np.log(flow.U_edges), np.log(flow.R_edges) )

fig, ax = plt.subplots()
ax.plot(flow.R_edges - 1, dlnU_dlnR, label="finite difference gradient")
ax.plot(flow.R_edges - 1, np.sqrt(0.5) * (flow.R_edges - 1)**-0.5, linestyle="dotted", lw=3, label="$(2 (R - 1))^{-1/2}$")
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel("$(R - 1)$")
ax.set_ylabel(r"$\dfrac{d \, \ln U}{d \, \ln R}$")
ax.legend()

# That looks very close to a power law in $(R - 1)$ (see orange dotted line, although not exactly).
#
# So we now have all we need to calculate $H_n$

Hn = flow.R_edges / 2 / flow.U_edges / (1 + dlnU_dlnR)

# But we also need to normalize it so that it integrates to unity. 

Hn0 = np.trapezoid(Hn, flow.density_edges)
abs(Hn0)

fig, ax = plt.subplots()
ax.plot(flow.density_edges, Hn / abs(Hn0), label="Dyson (1968)")
ax.plot(flow.density_edges, 0.5 / np.sqrt(flow.density_edges), label="Constant")
ax.set_ylim(0, 3.0)
ax.set_xlabel("Dimensionless density, $n$")
ax.set_ylabel("$H_n$")
ax.legend().set_title("Velocity law")
sns.despine()

# Finally, we can convert to log space and find the Density Energy Distribution: $n H_n$

# +
fig, ax = plt.subplots()
ax.plot(flow.density_edges, flow.density_edges * Hn / abs(Hn0), label="Dyson (1968)")
ax.plot(flow.density_edges, 0.5 * np.sqrt(flow.density_edges), label="Constant")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(5e-3, 1.0)
ax.set_xlabel("Dimensionless density, $n$")
ax.set_ylabel(r"DED$\equiv n\, H_n$")
ax.legend().set_title("Velocity law")
ax.set_title(r"Single-globule Densimetric Energy Distribution", y=1.05)

sns.despine()
fig.savefig("single-globule-ded.pdf", bbox_inches="tight")
# -

# ### Possible change of notation
#
# If we are going to talk about the *Density Energy Distribution* then we should maybe call it 
# $$ 
# \frac{d E}{d \log_e n}
# $$
# where
# $$
# E = \int_{\mathrm{volume}} e\, dV
# \quad
# \mathrm{and}
# \quad
# e \equiv 4\pi j = h\nu \, \alpha_{\mathrm{eff}} \, n^2
# $$
# is luminosity for some recombination line, such as H$\alpha$ (assuming that internal extinction can be neglected). Or it could be free-free radio emission, so long as we are on the flat part of the radio spectrum (optically thin). 
#
# Then we could either use a dimensional form (with units of solar luminosities, for instance) or a non-dimensional form:
# $$
# \frac{1}{E} \frac{d E}{d \log_e n} = \frac{d \log_e E}{d \log_e n}
# $$

# ### Relation between DED and $s$-PDF and $n$-PDF
#
# For the "standard" log-density PDF I will take the fraction of the *volume* that is occupied by gas of a certain density: 
# $$
# \frac{d V}{d \log_e n}.
# $$
# In the turbulence literature, this is called the $s$-pdf, where $s = \log_e \rho / \rho_0$, where $\rho_0$ is the mean density over the volume.  If the density distribution is log-normal then the $s$-pdf is a Gaussian (but see Beattie:2022a for deviations from this). 
#
# Beattie et al study what they call the "volume- and mass-weighted logarithmic density-PDFs". So, in the same way we can call the DED an "luminosity-weighted logarithmic density-PDF". 
#
# Also note that the gaussian is not centered on $s=0$, but instead on $s = s_0 = -\sigma_s^2 / 2$, where $\sigma_s$ is the width of the gaussian in $s$-space.  And for our wind model, it is problematic to define $\rho_0$ since the volume of the wind is formally infinite. That is one of the reasons why we prefer to concentrate on the DED. 
#
#

# ## Calculate the density histogram

# This was my first attempt to calculate the PDF, using the histogram method. It gives the same answer as the analytic calculation above (within the limitations of the discrete sampling). 

H, den_bin_edges = flow.radial_density_histogram(flow.R_edges, flow.density_edges, nbins=100)

# First, we do it on a linear scale, calculating $H_n$

fig, ax = plt.subplots()
ax.stairs(H, den_bin_edges)
ax.set_xlabel("Dimensionless Density, $n$")
ax.set_ylabel("Histogram, $H_n$")

# Now calculate the Density Energy Distribution on log scale. The value of the DED is the contribution of each density to the total recombination luminosity of the flow.  It is normalized such that 
# $$
# \int_{n = 0}^1 \mathrm{DED}(n) \, d \log_e n = 1
# $$
#

# +
fig, ax = plt.subplots()
den_bin_centers = 0.5*(den_bin_edges[1:] + den_bin_edges[:-1])

ax.stairs(H * den_bin_centers, den_bin_edges, label="Dyson (1968)")
ax.plot(den_bin_centers, 0.5 * den_bin_centers ** 0.5, label="Constant")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(None, 1.0)
ax.legend().set_title("Velocity law")
ax.set_title(r"Single-globule Density Energy Distribution", y=1.05)
ax.set_xlabel("Dimensionless density, $n$")
ax.set_ylabel(r"DED$\equiv n \,H_n$")
sns.despine()
# -

# This looks more reasonable now.  The orange line is the constant velocity result

# #



# ## Test the integration of emissivity

np.trapzd(flow.R_edges







flow.alpha

radii, densities = flow.R_edges, flow.density_edges

import numpy as np

radii_centers = 0.5 * (radii[:-1] + radii[1:])
volume_elements = radii_centers**flow.alpha * np.diff(radii)
density_centers = 0.5 * (densities[:-1] + densities[1:])
weights = density_centers**2 * volume_elements


radii_centers


