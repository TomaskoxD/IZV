#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: Tomas Ondrusek (xondru18)

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene na prednasce
"""


from bs4 import BeautifulSoup
import requests
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def integrate(x: np.array, y: np.array) -> float:
    """Integrate function using trapezoidal rule

    Args:
        x (np.array): x values
        y (np.array): y values

    Returns:
        float: integral value
    """
    return np.sum(np.multiply(np.subtract(x[1:], x[:-1]), np.add(y[:-1], y[1:]) / 2))


def generate_graph(
    a: List[float], show_figure: bool = False, save_path: str | None = None
):
    """Generate graph from list of floats

    Args:
        a (List[float]): list of floats
        show_figure (bool, optional): show figure. Defaults to False.
        save_path (str | None, optional): path to save figure. Defaults to None.
    """
    fig = plt.figure(figsize=(15, 6))
    ax = plt.subplot()

    x = np.linspace(-3, 3, 1000)
    y = np.multiply(np.power(x, 2).reshape(1000, 1), np.array(a).reshape(1, 3))

    for i in range(3):
        ax.plot(x, y[:, i], label="$y_{%.1f}(x)$" % (a[i]))

    ax.fill_between(x, y[:, 0], alpha=0.1, color="blue")
    ax.fill_between(x, y[:, 1], alpha=0.1, color="orange")
    ax.fill_between(x, y[:, 2], alpha=0.1, color="green")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_xlabel("x")
    ax.set_ylabel("$f_{a}(x)$")
    ax.set_label("f(x)")
    # Put a legend to the top part of current plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3)
    ax.text(3, 17.5, r"$\int_{f_{2.0}}(x)dx$", fontsize=12)
    ax.text(3, 9.5, r"$\int_{f_{1.0}}(x)dx$", fontsize=12)
    ax.text(3, -19, r"$\int_{f_{-2.0}}(x)dx$", fontsize=12)
    plt.xlim([-3, 3.7])
    plt.ylim([-20, 20])
    if show_figure:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close("all")


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    """Generate tree sinus graphs

    Args:
        show_figure (bool, optional): show plotted graphs. Defaults to False.
        save_path (str | None, optional): save plotted graphs. Defaults to None.
    """
    _, axes = plt.subplots(3, 1, figsize=(15, 12))

    x = np.linspace(0, 100, 8000)
    y1 = np.multiply(np.sin(x / 50 * np.pi), 0.5)
    y2 = np.multiply(np.sin(x * np.pi), 0.25)
    y3 = np.add(y1, y2)

    for i in range(3):
        axes[i].set_xlabel("t")
        axes[i].set_ylim([-0.8, 0.8])
        axes[i].set_xlim([0, 100])
        axes[i].set_yticks([-0.8, -0.4, 0, 0.4, 0.8])

    axes[0].set_ylabel("$f_{1}(x)$")
    axes[0].plot(x, y1)

    axes[1].set_ylabel("$f_{2}(x)$")
    axes[1].plot(x, y2)

    axes[2].set_ylabel("$f_{1}(x) + f_{2}(x)$")
    axes[2].plot(x, np.ma.masked_less(y3, y1), "g")
    axes[2].plot(x, np.ma.masked_greater(y3, y1), "r")

    # show all subplots
    if show_figure:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close("all")


def download_data(url="https://ehw.fit.vutbr.cz/izv/temp.html"):
    """Download data from url and return it as a list

    Args:
        url (str, optional): string containing url to download data from . Defaults to "https://ehw.fit.vutbr.cz/izv/temp.html".

    Returns:
        data: downloaded data from url in format of list of tuples
    """

    data = requests.get(url)
    soup = BeautifulSoup(data.text, "html.parser")
    table = soup.find("table")
    rows = table.find_all("tr")
    data = []
    for row in rows:
        cols = row.find_all("td")
        cols = [x.text.strip() for x in cols]

        if len(cols) > 0:
            temp = np.array(cols[2:])
            # replace comma with dot
            temp = np.char.replace(temp, ",", ".")
            # create mask for string values that contain valid float
            mask = np.char.count(temp, ".") == 1
            # remove invalid values
            temp = temp[mask]

            data.append(
                {
                    "year": int(cols[0]),
                    "month": int(cols[1]),
                    "temp": temp.astype(float),
                }
            )
    return data


def get_avg_temp(data, year=None, month=None) -> float:
    """Get average temperature from data

    Args:
        data (list): list of tuples containing data
        year (int, optional): year to get average temperature from. Defaults to None.
        month (int, optional): month to get average temperature from. Defaults to None.

    Returns:
        float: average temperature
    """
    if year is None and month is None:
        return np.mean([np.mean(d["temp"]) for d in data])
    elif year is not None and month is None:
        return np.mean(np.concatenate([x["temp"] for x in data if x["year"] == year]))
    elif year is None and month is not None:
        return np.mean(np.concatenate([x["temp"] for x in data if x["month"] == month]))
    else:
        return np.mean(
            np.concatenate(
                [x["temp"] for x in data if x["year"] == year and x["month"] == month]
            )
        )
