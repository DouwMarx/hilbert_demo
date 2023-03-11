import dash_core_components as dcc
import dash_html_components as html
import dash
import urllib.parse
import re


def convert(text):
    def toimage(x):
        if x[1] and x[-2] == r'$':
            x = x[2:-2]
            img = "\n<img src='https://math.now.sh?from={}' style='display: block; margin: 0.5em auto;'>\n"
            return img.format(urllib.parse.quote_plus(x))
        else:
            x = x[1:-1]
            return r'![](https://math.now.sh?from={})'.format(urllib.parse.quote_plus(x))
    return re.sub(r'\${2}([^$]+)\${2}|\$(.+?)\$', lambda x: toimage(x.group()), text)


app = dash.Dash()

Markdown_text = r"""
Let's see if it works:  
$$\hat P \psi_k(x) =p \psi_k(x)$$ 

$$-i\hbar \frac{\partial {c\ e^{ikx}}}{\partial x} =-i\hbar\ c\ ik\ e^{ikx} $$ 

$$\hbar k\ c\ e^{ikx} = \hbar k\ \psi_k(x) \tag{2}$$
with $p=\hbar k$
"""

Markdown_text = convert(Markdown_text)


app.layout = html.Div([
    dcc.Markdown(Markdown_text, dangerously_allow_html=True)
])

if __name__ == '__main__':
    app.run_server(host='0.0.0.0')