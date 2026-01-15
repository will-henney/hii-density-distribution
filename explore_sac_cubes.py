import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from astropy.io import fits
    import numpy as np
    from matplotlib import pyplot as plt
    import seaborn as sns
    from pathlib import Path
    return Path, fits, mo


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
def _(DATAPATH, fits):
    def open_cube(
        time: int,
        suffix: str,
        prefix: str = "04052012_4",
    ):
        hdulist = fits.open(
            DATAPATH
            / f"{prefix}_{time:04d}{suffix}.fits"
        )
        return hdulist[0].data


    class Cube:
        def __init__(self, time: int):
            self.d = open_cube(time, "d")
            self.xn = open_cube(time, "x")
            self.xi = 1.0 - self.xn
            self.di = self.d * self.xi
    return (Cube,)


@app.cell
def _(mo):
    TIME = mo.ui.number(
        3,
        30,
        1,
        label="Simulation time (kyr): ",
    )
    TIME
    return (TIME,)


@app.cell
def _(Cube, TIME):
    cube = Cube(TIME.value)
    cube.di.max()
    return


@app.cell
def _(TIME):
    TIME.value
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
