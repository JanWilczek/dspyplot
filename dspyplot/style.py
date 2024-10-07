from matplotlib import rc
import matplotlib.pyplot as plt

stem_params = {"linefmt": "C0-", "markerfmt": "C0o", "basefmt": "k"}
rc("font", **{"family": "sans-serif", "sans-serif": ["Verdana"]})
fontsize = 20
plt.rcParams.update({"font.size": fontsize})
color = "#ef7600"
complementary_color_1 = "#0008ef"
complementary_color_2 = "#7400ef"
complementary_color_3 = "#00efe7"
tetradic_color_2 = "#eb00ef"
tetradic_color_3 = "#efef00"
triadic_color_1 = "#04ef00"
window_color = complementary_color_1
grey = "#7c7c7c"
red = "#ef0001"  # https://www.canva.com/colors/color-wheel/
save_params = {"dpi": 300, "bbox_inches": "tight", "transparent": True}
img_file_suffix = ".png"
color_palette = [
    color,
    complementary_color_1,
    triadic_color_1,
    tetradic_color_2,
    tetradic_color_3,
    complementary_color_2,
    complementary_color_3,
    grey,
]
color_palette_names = [
    "color",
    "complementary_color_1",
    "triadic_color_1",
    "tetradic_color_2",
    "tetradic_color_3",
    "complementary_color_2",
    "complementary_color_3",
    "grey",
]


def plot_color_palette():
    # Create the plot
    _, ax = plt.subplots(figsize=(12, 6))

    # Plot each color as a square
    for i, color in enumerate(color_palette):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))

    # Set limits and labels
    ax.set_xlim(0, len(color_palette))
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(color_palette)))
    color_names = [
        f"{name}:\n{hex}" for name, hex in zip(color_palette_names, color_palette)
    ]
    ax.set_xticklabels(color_names, rotation=45, ha="center")
    ax.set_yticks([])

    # Add a title
    ax.set_title("WolfSound's color palette")
    plt.subplots_adjust(bottom=0.45)

    # Show the plot
    plt.show()


def main():
    plot_color_palette()


if __name__ == "__main__":
    main()
