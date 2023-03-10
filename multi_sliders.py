import plotly.express as px
import plotly.graph_objects as go

df = px.data.gapminder()
# add another column to "animate" on...
df["decade-continent"] = df.apply(
    lambda x: f'{round(x["year"], -1)}-{x["continent"]}', axis=1
)

# use px to do heavy lifting construction
figs = [
    px.scatter(
        df,
        x="gdpPercap",
        y="lifeExp",
        animation_frame=ac,
    )
    # columns that become sliders
    for ac in ["continent", "year", "decade-continent"]
]

# extract frames and sliders from each of the animated figures
layout = figs[0].to_dict()["layout"]
layout.pop("updatemenus") # don't want play and pause buttons
layout["sliders"] = []
frames = []
for i, f in enumerate(figs):
    slider = f.to_dict()["layout"]["sliders"]
    slider[0]["y"] = -0.6 * i
    slider[0]["x"] = 0
    slider[0]["len"] = 1

    layout["sliders"] += slider
    frames += f.frames

figs.show()