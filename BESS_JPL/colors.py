from matplotlib.colors import LinearSegmentedColormap

GPP_COLORMAP = LinearSegmentedColormap.from_list(
    name="GPP",
    colors=[
        "#000000",
        "#bdae08",
        "#325e32",
        "#a6ff01",
        "#00ff00"
    ]
)

ET_COLORMAP = LinearSegmentedColormap.from_list("ET", [
    "#f6e8c3",
    "#d8b365",
    "#99974a",
    "#53792d",
    "#6bdfd2",
    "#1839c5"
])

NDVI_COLORMAP_ABSOLUTE = LinearSegmentedColormap.from_list(
    name="NDVI",
    colors=[
        (0, "#0000ff"),
        (0.4, "#000000"),
        (0.5, "#745d1a"),
        (0.6, "#e1dea2"),
        (0.8, "#45ff01"),
        (1, "#325e32")
    ]
)
