import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Density PDF from turbulent H II region simulations

    We will use the cubes that Sac produced in 2012
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## User parameters
    """)
    return


@app.cell
def _(mo):
    TIME = mo.ui.number(
        3,
        30,
        1,
        label="Simulation time (kyr): ",
    )
    MIN_IFRAC = mo.ui.number(
        0,
        1,
        1e-4,
        0.01,
        label="Minimum ionization fraction",
    )
    TIME, MIN_IFRAC
    return MIN_IFRAC, TIME


@app.cell
def _(MIN_IFRAC, cube, d_max, dmean_e, dmean_v, drms_v, dsig_v, mo):
    mo.md(rf"""
    * Mean density over volume = {dmean_v:.0f} +/- {dsig_v:.0f}
    * RMS density = {drms_v:.0f}   
    * Emission-weighted mean density = {dmean_e:.0f}  
    * Maximum density = {d_max:.0f}
    * Fraction of cube with ionization fraction > {MIN_IFRAC.value:.2g} = {cube.vfrac:.5f}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Plots
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Matplotlib version of 1d PDFs. First the density
    """)
    return


@app.cell
def _(cube, dmean_e, dmean_v, drms_v, np, plt, sns):
    bins = np.geomspace(1.0, 1.0e5, 100)
    _fig, _ax = plt.subplots()
    _H, _ = np.histogram(
        cube.di_m,
        weights=cube.e_m,
        bins=bins,
        density=True,
    )
    _ax.stairs(
        _H * bins[:-1], bins, fill=True
    )
    _
    _ax.axvline(dmean_v, color="k")
    _ax.axvline(drms_v, color="k")
    _ax.axvline(dmean_e, color="k")
    _ax.set_xscale("log")
    _ax.set_yscale("log")
    _ax.set_xlabel(
        r"Ionized density, cm$^{-3}$"
    )
    _ax.set_ylabel(r"Probability density")
    sns.despine()
    _fig
    return


@app.cell
def _():
    # sns.histplot(data={"x": cube.di_m, "w": cube.e_m}, x="x", weights="w", log_scale=True)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Then the ionization parameter
    """)
    return


@app.cell
def _(cube, np, plt, sns):
    Ubins = np.geomspace(1e-4, 1.0e8, 100)
    _fig, _ax = plt.subplots()
    _H, _ = np.histogram(
        cube.U_m,
        weights=cube.e_m,
        bins=Ubins,
        density=True,
    )
    _ax.stairs(
        _H * Ubins[:-1], Ubins, fill=True
    )
    _ax.set_xscale("log")
    _ax.set_yscale("log")
    _ax.set_xlabel(r"Ionization parameter")
    _ax.set_ylabel(r"Probability density")
    sns.despine()
    _fig
    return


@app.cell
def _(Cube, cube, dmean_e, np, plt):
    def plot_hist2d(
        c: Cube,
        drange: tuple[float, float] = [
            0.0,
            5.0,
        ],
        Urange: tuple[float, float] = [
            -4.0,
            8.0,
        ],
        bins: int = 50,
        gamma: float = 2.0,
    ):
        HH, _, _ = np.histogram2d(
            x=np.log10(c.di_m),
            y=np.log10(c.U_m),
            bins=bins,
            range=[drange, Urange],
            density=True,
            weights=c.e_m,
        )
        fig = plt.figure()
        ax = fig.add_subplot()
        # fig, ax = plt.subplots() # This is worse since type of ax cannot be inferred
        ax.imshow(
            HH.T ** (1 / gamma),
            origin="lower",
            aspect="auto",
            extent=[*drange, *Urange],
            cmap="magma_r",
        )
        # ax.contour(HH.T, extent=[*drange, *Urange])
        ax.axvline(
            np.log10(dmean_e),
            c="c",
            ls="dotted",
        )
        ax.axhline(0.0, c="c", ls="dotted")
        ax.set_xlabel(r"log$_{10}$ density")
        ax.set_ylabel(
            r"log$_{10}$ ionization parameter"
        )
        return fig


    plot_hist2d(cube, gamma=3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Define the cube
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Read the FITS files for the specified simulation time
    """)
    return


@app.cell
def _(Cube, MIN_IFRAC, TIME):
    cube = Cube(
        TIME.value,
        min_ifrac=MIN_IFRAC.value,
    )
    return (cube,)


@app.cell
def _(mo):
    mo.md(r"""
    Calculate some statistics
    """)
    return


@app.cell
def _(cube, np):
    dmean_v = np.mean(cube.di_m)
    dmean_e = np.average(
        cube.di_m, weights=cube.e_m
    )
    d_max = np.max(cube.di_m)
    dsig_v = np.std(cube.di_m)
    drms_v = np.sqrt(np.mean(cube.di_m**2))
    return d_max, dmean_e, dmean_v, drms_v, dsig_v


@app.cell
def _(Path):
    DATAPATH = (
        Path.home()
        / "Work"
        / "Garrelt"
        / "results-sac"
    )
    DATAPATH
    return (DATAPATH,)


@app.cell
def _(mo):
    mo.md(r"""
    Functions and classes for reading the physical variables for the simulation cube
    """)
    return


@app.cell
def _(DATAPATH, fits, np):
    def open_cube(
        time: int,
        suffix: str,
        prefix: str = "04052012_4",
    ):
        hdulist = fits.open(
            DATAPATH
            / f"{prefix}_{time:04d}{suffix}.fits"
        )
        return hdulist[0].data.astype(
            "float"
        )


    class Cube:
        def __init__(
            self, time: int, min_ifrac=0.01
        ):
            self.d = open_cube(time, "d")
            self.xn = open_cube(time, "x")
            self.e = open_cube(
                time, "e-Halpha"
            )
            self.xi = 1.0 - self.xn
            self.di = self.d * self.xi
            mask = self.xi >= min_ifrac
            self.di_m = self.di[mask]
            self.e_m = self.e[mask]
            self.vfrac = (
                np.sum(mask) / mask.size
            )
            # Ionization parameter
            self.U = self.xi**2 / self.xn
            self.U_m = self.U[mask]
    return (Cube,)


@app.cell
def _(cube):
    cube.vfrac
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell
def _():
    import marimo as mo
    from astropy.io import fits
    import numpy as np
    from matplotlib import pyplot as plt
    import seaborn as sns
    from pathlib import Path
    return Path, fits, mo, np, plt, sns


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
