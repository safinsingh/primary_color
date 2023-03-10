from PIL import Image
import pandas as pd
import numpy as np
import math
import webcolors
import os


def is_pixel_greyscale(df):
    return (df["red"] == df["green"]) & (df["green"] == df["blue"])


def is_greyscale(df: pd.DataFrame):
    return len(df.loc[is_pixel_greyscale(df)]) == len(df)


def closest(rgb):
    rgb_colors = [
        (np.array(webcolors.hex_to_rgb(hex_color)), name)
        for (hex_color, name) in webcolors.CSS3_HEX_TO_NAMES.items()
    ]
    differences = [
        (np.linalg.norm(rgb_color - np.array(rgb)), name)
        for (rgb_color, name) in rgb_colors
    ]
    min_difference = min(differences, key=lambda d: d[0])
    return min_difference[1]


def primary_color(source):
    im = Image.open(source)
    im_data = np.array(im.getdata()).reshape(-1, 3)
    xs, ys = map(lambda s: s.reshape(-1, 1), np.indices((im.height, im.width)))

    im_df = pd.DataFrame(
        np.hstack((xs, ys, im_data)), columns=["x", "y", "red", "green", "blue"]
    )

    if is_greyscale(im_df):
        print("[-] cannot determine primary color of greyscale image")
        exit(1)

    im_df.drop(
        im_df[is_pixel_greyscale(im_df)].index,
        inplace=True,
    )

    bin_edges = np.arange(0, 256, 17)
    red_cats = pd.cut(im_df["red"], bin_edges)
    green_cats = pd.cut(im_df["green"], bin_edges)
    blue_cats = pd.cut(im_df["blue"], bin_edges)

    im_ranges = im_df.groupby([red_cats, green_cats, blue_cats]).size()
    im_ranges = im_ranges.sort_values(ascending=False)

    rgb = [math.ceil(i.mid) for i in im_ranges.index[0]]
    return closest(rgb)


res_path = "res"
for f in os.listdir(res_path):
    abs_path = os.path.join(res_path, f)
    color = primary_color(abs_path)
    print("The primary color of `" + f + "` is " + color)
