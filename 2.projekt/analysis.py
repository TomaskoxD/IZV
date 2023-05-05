#!/usr/bin/env python3.9
# coding=utf-8

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import zipfile

regions_map = {
    "PHA": "00",
    "STC": "01",
    "JHC": "02",
    "PLK": "03",
    "ULK": "04",
    "HKK": "05",
    "JHM": "06",
    "MSK": "07",
    "OLK": "14",
    "ZLK": "15",
    "VYS": "16",
    "PAK": "17",
    "LBK": "18",
    "KVK": "19",
}
# swap keys and values
regions_map = {v: k for k, v in regions_map.items()}
my_regions = ["PHA", "STC", "JHM", "PLK"]

# muzete pridat libovolnou zakladni knihovnu ci knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz

# Ukol 1: nacteni dat ze ZIP souboru
def load_data(filename: str) -> pd.DataFrame:
    # tyto konstanty nemente, pomuzou vam pri nacitani
    headers = [
        "p1",
        "p36",
        "p37",
        "p2a",
        "weekday(p2a)",
        "p2b",
        "p6",
        "p7",
        "p8",
        "p9",
        "p10",
        "p11",
        "p12",
        "p13a",
        "p13b",
        "p13c",
        "p14",
        "p15",
        "p16",
        "p17",
        "p18",
        "p19",
        "p20",
        "p21",
        "p22",
        "p23",
        "p24",
        "p27",
        "p28",
        "p34",
        "p35",
        "p39",
        "p44",
        "p45a",
        "p47",
        "p48a",
        "p49",
        "p50a",
        "p50b",
        "p51",
        "p52",
        "p53",
        "p55a",
        "p57",
        "p58",
        "a",
        "b",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "p5a",
    ]
    # def get_dataframe(filename: str, verbose: bool = False) -> pd.DataFrame:
    regions = {
        "PHA": "00",
        "STC": "01",
        "JHC": "02",
        "PLK": "03",
        "ULK": "04",
        "HKK": "05",
        "JHM": "06",
        "MSK": "07",
        "OLK": "14",
        "ZLK": "15",
        "VYS": "16",
        "PAK": "17",
        "LBK": "18",
        "KVK": "19",
    }

    df = pd.DataFrame()
    with zipfile.ZipFile(filename, "r") as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith(".zip"):
                with zipfile.ZipFile(zip_ref.open(file), "r") as zip_ref2:
                    for file2 in zip_ref2.namelist():
                        if file2.endswith(".csv"):
                            if file2 == "CHODCI.csv":
                                continue
                            if zip_ref2.getinfo(file2).file_size == 0:
                                continue
                            dataframe = pd.read_csv(
                                zip_ref2.open(file2),
                                encoding="cp1250",
                                delimiter=";",
                                low_memory=False,
                            )
                            dataframe.columns = headers
                            dataframe["region"] = file2[0:2]
                            df = pd.concat([df, dataframe])
    df["region"] = df["region"].replace(regions_map)
    return df


# Ukol 2: zpracovani dat
def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    df2 = df.copy()
    orig_size = round(df2.memory_usage(index=False, deep=True).sum() / 1000000, 1)
    # columns to be changed to category and float
    category_cols = ["k", "p", "q", "t", "l", "i", "h"]
    float_cols = ["a", "b", "d", "e", "f", "g", "n", "o"]

    # change columns to category and float
    df2[category_cols] = df2[category_cols].astype("category")
    # replace commas with dots and change to float
    df2[float_cols] = df2[float_cols].replace("[^0-9]", np.nan, regex=True)
    df2[float_cols] = df2[float_cols].replace(",", ".", regex=True)
    df2[float_cols] = df2[float_cols].astype("float64")

    # change date to datetime
    df2.rename(columns={"p2a": "date"}, inplace=True)
    df2["date"] = pd.to_datetime(df2["date"])

    # drop duplicates
    df2 = df2.drop_duplicates(subset=["p1"], keep="first")
    new_size = round(df2.memory_usage(index=False, deep=True).sum() / 1000000, 1)

    if verbose:
        print(f"orig_size={orig_size} MB")
        print(f"new_size={new_size} MB")
    return df2


# Ukol 3: počty nehod v jednotlivých regionech podle viditelnosti
def plot_visibility(
    df: pd.DataFrame, fig_location: str = None, show_figure: bool = False
):
    # get data
    pd.melt(df, id_vars=["region"], value_vars=["p19"])

    data = df[df["region"].isin(my_regions)]
    data = data.groupby(["region", "p19"]).size().reset_index(name="count")
    data = data.pivot(index="region", columns="p19", values="count")
    data = data.fillna(0)
    data = data.astype(int)

    data[2] = data[2] + data[3]
    data[4] = data[4] + data[6]
    data[5] = data[5] + data[7]
    data = data.drop([3, 6, 7], axis=1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle("Počet nehod dle viditelnosti", fontsize=12)
    fig.subplots_adjust(hspace=0.2, top=0.91, bottom=0.05)
    fig.patch.set_facecolor("#FFFFFF")

    den_nezhorsena = data[1]
    den_zhorsena = data[2]
    noc_nezhorsena = data[4]
    noc_zhorsena = data[5]

    # plots
    for i, ax in enumerate(fig.axes):
        if i == 0:
            ax = sns.barplot(x=noc_nezhorsena.index, y=noc_nezhorsena.values, ax=ax1)
            ax.set_title("Viditelnost: v noci - nezhoršená", fontsize=10)
        elif i == 1:
            ax = sns.barplot(x=noc_zhorsena.index, y=noc_zhorsena.values, ax=ax2)
            ax.set_title("Viditelnost: v noci - zhoršená", fontsize=10)
        elif i == 2:
            ax = sns.barplot(x=den_nezhorsena.index, y=den_nezhorsena.values, ax=ax3)
            ax.set_title("Viditelnost: ve dne - nezhoršená", fontsize=10)
        elif i == 3:
            ax = sns.barplot(x=den_zhorsena.index, y=den_zhorsena.values, ax=ax4)
            ax.set_title("Viditelnost: ve dne - zhoršená", fontsize=10)
        ax.set_ylabel("Počet nehod")
        ax.set_facecolor("#cccccc")
        ax.grid(color="#000000", linestyle="-", linewidth=1, axis="y", alpha=0.3)

        # remove borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Kraj")
        ax.set_xticklabels([x for x in my_regions])
        ax.tick_params(left=False, bottom=False)
        ax.set_alpha(0.9)

    if show_figure:
        plt.show()

    if fig_location is not None:
        plt.savefig(fig_location, dpi=600, bbox_inches="tight")
    plt.close()


# Ukol4: druh srážky jedoucích vozidel
def plot_direction(
    df: pd.DataFrame, fig_location: str = None, show_figure: bool = False
):
    df1 = [None] * 4

    # get data for each region and melt it to long format for plotting with seaborn barplot
    for i, region in enumerate(my_regions):
        df1[i] = df[df["region"] == region].copy()
        df1[i]["month"] = df1[i]["date"].dt.month
        df1[i] = df1[i].groupby(["month", "p7"])["p7"].count().unstack().sort_index()
        df1[i][2] = df1[i][2] + df1[i][3]
        df1[i] = df1[i].drop(columns=[3])
        df1[i] = df1[i].drop(columns=[0])
        df1[i] = df1[i].rename(columns={2: "boční", 4: "zezadu", 1: "čelní"})
        df1[i] = pd.melt(df1[i].reset_index(), id_vars=["month"])
        df1[i] = df1[i].rename(
            columns={
                "month": "Měsíc",
                "variable": "směr",
                "value": "Počet nehod",
                "p7": "Druh srážky",
            }
        )
        df1[i]["Druh srážky"] = pd.Categorical(
            df1[i]["Druh srážky"], categories=["boční", "zezadu", "čelní"], ordered=True
        )

    fig, ((graph1, graph2), (graph3, graph4)) = plt.subplots(2, 2, figsize=(20, 10))
    plt.subplots_adjust(hspace=0.3)
    # plot 1
    graph1 = sns.barplot(
        x="Měsíc", y="Počet nehod", hue="Druh srážky", data=df1[0], ax=graph1
    )
    graph1.set_title("Kraj: {}".format(my_regions[0]))
    graph1.legend_.remove()

    # plot 2
    graph2 = sns.barplot(
        x="Měsíc", y="Počet nehod", hue="Druh srážky", data=df1[1], ax=graph2
    )
    graph2.set_title("Kraj: {}".format(my_regions[1]))
    graph2.legend_.remove()

    # plot 3
    graph3 = sns.barplot(
        x="Měsíc", y="Počet nehod", hue="Druh srážky", data=df1[2], ax=graph3
    )
    graph3.set_title("Kraj: {}".format(my_regions[2]))
    graph3.legend_.remove()

    # plot 4
    graph4 = sns.barplot(
        x="Měsíc", y="Počet nehod", hue="Druh srážky", data=df1[3], ax=graph4
    )
    graph4.set_title("Kraj: {}".format(my_regions[3]))
    graph4.legend_.remove()

    # remove borders
    for i, ax in enumerate(fig.axes):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    plt.legend(
        bbox_to_anchor=(1.05, 1.3),
        loc=2,
        borderaxespad=0.0,
        ncol=1,
        title="Druh srážky",
        frameon=True,
        facecolor="white",
        edgecolor="white",
    )

    if show_figure:
        plt.show()

    if fig_location is not None:
        plt.savefig(fig_location, dpi=600, bbox_inches="tight")
    plt.close()


# Ukol 5: Následky v čase
def plot_consequences(
    df: pd.DataFrame, fig_location: str = None, show_figure: bool = False
):
    # get data
    df1 = df.copy()
    labels = ["Lehké zranení", "Težké zranění", "Usmrcení"]

    # create new column with consequences of accident and cut it to 3 categories
    df1["conseq"] = df1["p13c"] + df1["p13b"] + df1["p13a"]
    df1["conseq"] = pd.cut(df1["conseq"], [i for i in range(4)], labels=labels)

    data = df1[
        df1["region"].isin(my_regions)
        & (df1["p13c"] != 0)
        & (df1["date"].dt.year < 2022)
    ]

    # pivot table to get data for every region and every consequence
    data = pd.pivot_table(
        data, columns=["conseq"], values="p1", index=["region", "date"], aggfunc="count"
    )

    # resample data to get data for every month
    target = data.stack(level="conseq").unstack(level="region")

    for i, region in enumerate(my_regions):
        target[region] = data.loc[region].resample("M").sum().stack(level="conseq")

    # reset index to get data for every month
    target = target.stack(level="region").reset_index()

    fig, axs = plt.subplots(2, 2, figsize=(20, 10))

    # plot for every region
    sns.set_style("darkgrid")
    sns.despine(top=True, bottom=True, left=True, right=True)
    for i, region in enumerate(my_regions):
        ax = axs[i // 2, i % 2]
        sns.lineplot(
            data=target[target["region"] == region], x="date", y=0, hue="conseq", ax=ax
        )
        ax.set_title(f"Kraj: {region}")
        ax.set_xlabel("Kraj")
        if i % 2 == 0:
            ax.set_ylabel("Počet nehôd")
        else:
            ax.set_ylabel("")
        ax.legend_.remove()
        ax.set_xticks([f"20{year}-01" for year in range(16, 23)])
        ax.set_xticklabels([f"01/{year}" for year in range(16, 23)])
        ax.set_ylim(0, 300)
        ax.set_facecolor("#EAEAF2")

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Následky",
        loc="center right",
        facecolor="white",
        edgecolor="white",
    )

    if show_figure:
        plt.show()

    if fig_location is not None:
        plt.savefig(fig_location, dpi=600, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # zde je ukazka pouziti, tuto cast muzete modifikovat podle libosti
    # skript nebude pri testovani pousten primo, ale budou volany konkreni
    # funkce.

    df = load_data("data/data.zip")
    df2 = parse_data(df, True)
    plot_visibility(df2, "01_visibility.png")
    plot_direction(df2, "02_direction.png")
    plot_consequences(df2, "03_consequences.png")


# Poznamka:
# pro to, abyste se vyhnuli castemu nacitani muzete vyuzit napr
# VS Code a oznaceni jako bunky (radek #%%% )
# Pak muzete data jednou nacist a dale ladit jednotlive funkce
# Pripadne si muzete vysledny dataframe ulozit nekam na disk (pro ladici
# ucely) a nacitat jej naparsovany z disku
