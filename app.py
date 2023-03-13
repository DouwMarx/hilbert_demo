import numpy as np
from dash import Dash, dcc, html, Input, Output, State
from source import HilbertDemo
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

hd = HilbertDemo()


def make_collapsable_graph(graph_id, graph_title):
    return html.Div(
        [
            dbc.Button(
                graph_title,
                id=graph_id + "-collapse-button",
                className="mb-3",
                color="primary",
                n_clicks=0,
            ),
            dbc.Collapse(
                dcc.Graph(id=graph_id),
                id=graph_id + "-collapse",
                is_open=True,
            ),
        ]
    )


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

markdown_file = open("text.md")


app.layout = html.Div([
    html.Div(sliders +
             [
                 make_collapsable_graph('time-series-components', 'Time Series of Signal Components and Modulated Signal'),
        make_collapsable_graph('frequency-response', 'Frequency Response of Signal Components and Modulated Signal'),
        make_collapsable_graph('hilbert-time-response', 'Filtered and Unfiltered Signal Envelopes'),
        make_collapsable_graph('recovered-modulating-frequency-response', 'Frequency Response of Recovered and True Modulating Signal (DC Component Removed)    '),
                 dcc.Markdown(markdown_file.read(), mathjax=True),
             ]
             )]
)


# Add a callback for updating all plots using the sliders
@app.callback(
    [Output('time-series-components', 'figure'),
     Output('frequency-response', 'figure'),
     Output('hilbert-time-response', 'figure')],
    Output('recovered-modulating-frequency-response', 'figure'),
    [Input(slider_dict['id'], 'value') for slider_dict in hd.parameter_grid.values()]
)
def update_plots(*args):
    hd.update_state(*args)
    time_series_components = hd.make_time_domain_signal_components_plot()
    frequency_response = hd.make_frequency_domain_signal_components_plot()
    hilbert_time_response = hd.make_processed_time_series_plot()
    recovered_modulating_frequency_response = hd.make_processed_frequency_domain_plot()

    return time_series_components, frequency_response, hilbert_time_response, recovered_modulating_frequency_response


# Make the callbacks for collapsing the plots
for graph_id in ['time-series-components', 'frequency-response', 'hilbert-time-response','recovered-modulating-frequency-response']:
    @app.callback(
        Output(graph_id + "-collapse", "is_open"),
        Input(graph_id + "-collapse-button", "n_clicks"),
        State(graph_id + "-collapse", "is_open"),
    )
    def toggle_collapse(n, is_open):
        if n:
            return not is_open
        return is_open

if __name__ == '__main__':
    app.run_server(debug=True)
