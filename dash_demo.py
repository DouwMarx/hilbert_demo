import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

from discretized_modulation import HilbertDemo

app = Dash(__name__)

hd = HilbertDemo()

def make_slider_and_label_from_dict(slider_dict):
    slider = dcc.Slider(
        min=slider_dict['min'],
        max=slider_dict['max'],
        step=slider_dict['step'],
        id=slider_dict['id'],
        value=slider_dict['value'],
        marks={str(s1): str(s1) for s1 in np.linspace(slider_dict['min'], slider_dict['max'], 10).astype(int)},
    )
    label = html.Label(slider_dict['id'])
    return html.Div([label, slider])

sliders = [make_slider_and_label_from_dict(slider_dict) for slider_dict in hd.parameter_grid.values()]

app.layout = html.Div([
    html.Div( sliders + [
    dcc.Graph(id='time-series-components'),
    dcc.Graph(id='frequency-response'),
    ]
    )]
)


@app.callback(
    Output('time-series-components', 'figure'),
    Input(list(hd.parameter_grid.keys())[0], 'value'),
    Input(list(hd.parameter_grid.keys())[1], 'value')
    )
def update_time_graph(slider1, slider2):
    # Plot a sine wave with varying frequency and amplitude
    fig = hd.make_time_domain_signal_components_plot_at_state(slider1, slider2)
    return fig

@app.callback(
    Output('frequency-response', 'figure'),
    Input(list(hd.parameter_grid.keys())[0], 'value'),
    Input(list(hd.parameter_grid.keys())[1], 'value')
    )
def update_freq_graph(slider1, slider2):
    # Plot a sine wave with varying frequency and amplitude
    fig = hd.make_frequency_domain_signal_components_plot_at_state(slider1, slider2)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)