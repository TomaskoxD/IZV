#!/usr/bin/python3.10
# coding=utf-8
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily
import sklearn.cluster
import numpy as np

# muzete pridat vlastni knihovny


def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """Make a GeoDataFrame from a DataFrame with coordinates and date columns.

    Args:
        df (pd.DataFrame): dataframe

    Returns:
        geopandas.GeoDataFrame: geodataframe
    """

    # remove nan and inf values
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["d", "e"], how="all")

    # Make a date column
    df["date"] = pd.to_datetime(df["p2a"], cache=True)

    return geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(df["d"], df["e"]), crs="EPSG:5514"
    )


def plot_geo(
    gdf: geopandas.GeoDataFrame, fig_location: str = None, show_figure: bool = False
):
    """Plot a GeoDataFrame with contextily basemap.

    Args:
        gdf (geopandas.GeoDataFrame): geodataframe
        fig_location (str, optional): location to save figure. Defaults to None.
        show_figure (bool, optional): show figure. Defaults to False.
    """

    region = "JHM"

    data = (
        gdf[
            (gdf["region"] == region)
            & (gdf["p11"] >= 3)
            & gdf["date"].dt.year.isin([2018, 2019, 2020, 2021])
        ]
        .set_geometry(gdf.centroid)
        .to_crs("EPSG:3857")
    )

    # get graph bounds
    bounds = data.total_bounds
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    i = 0
    for r in range(2):
        for c in range(2):
            axs[r, c].set_title(region + " kraj ({})".format(2018 + i))
            axs[r, c].axis("off")
            # axs[r, c].set_xlim(xmin=1750000.0, xmax=bounds[2])
            # axs[r, c].set_ylim(ymin=bounds[1], ymax=bounds[3])

            data[(data["date"].dt.year == (2018 + i))].centroid.plot(
                ax=axs[r, c], markersize=1, color="red"
            )
            contextily.add_basemap(
                axs[r, c],
                crs=data.crs.to_string(),
                source=contextily.providers.Stamen.TonerLite,
            )
            i += 1

    plt.tight_layout()

    if fig_location:
        plt.savefig(fig_location)
    
    if show_figure:
        plt.show()


def plot_cluster(
    gdf: geopandas.GeoDataFrame, fig_location: str = None, show_figure: bool = False
):

    """Plot a GeoDataFrame with contextily basemap and clustered points.

    Args:
        gdf (geopandas.GeoDataFrame): geodataframe
        fig_location (str, optional): location to save figure. Defaults to None.
        show_figure (bool, optional): show figure. Defaults to False.
    """

    chosen_region = "JHM"
    title_str = f"Nehody v {chosen_region} kraji na silnicích 1., 2. a 3. třídy"

    # Subplots
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    # filter region, roadtype and transform to webmercator
    data = (
        gdf[(gdf["region"] == chosen_region) & gdf["p36"].isin([1, 2, 3])]
        .set_geometry(gdf.centroid)
        .to_crs("EPSG:3857")
    )
    gdf = gdf.set_geometry(gdf.centroid).to_crs("EPSG:3857")

    # collect points in a 2d array
    points = np.reshape(list(zip(data.geometry.x, data.geometry.y)), (-1, 2))

    # cluster points
    data["frequency_group"] = sklearn.cluster.MiniBatchKMeans(n_clusters=20).fit(points).labels_

    data = data.dissolve(by="frequency_group", aggfunc={"region": "count"})

    data.plot(
        ax=ax,
        markersize=1,
        column="region",
        legend=True,
        alpha=0.5,
        legend_kwds={"orientation": "horizontal", "label": "Počet nehod v úseku"},
    )
    ax.set_axis_off()
    contextily.add_basemap(
        ax,
        crs=data.crs.to_string(),
        alpha=0.9,
        reset_extent=False,
        source=contextily.providers.Stamen.TonerLite,
    )
    ax.set_title(title_str, fontsize="small")
    plt.tight_layout()

    if fig_location:
        plt.savefig(fig_location)

    if show_figure:
        plt.show()


if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    gdf = make_geo(pd.read_pickle("accidents.pkl.gz"))
    plot_geo(gdf, "geo1.png", True)
    plot_cluster(gdf, "geo2.png", True)
