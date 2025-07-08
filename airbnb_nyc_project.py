"""
airbnb_nyc_dashboard.py
────────────────────────────────────────────────────────────
Interactive Dash dashboard for Airbnb NYC 2019.
"""

#  Imports  #
import argparse, base64
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, dash_table
from dash.exceptions import PreventUpdate
from scipy import stats
from sklearn.decomposition import PCA

#  Data-clean helper  #
def clean_dataframe(csv="AB_NYC_2019.csv") -> pd.DataFrame:
    df = pd.read_csv(csv)

    # unique listings
    df = df.drop_duplicates("id").dropna(subset=["id"])
    df["id"] = df["id"].astype(int)

    # parse dates (keeps NaT if missing)
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    # safe fill for monthly reviews
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0).astype(float)

    return df.reset_index(drop=True)

df_clean = clean_dataframe()
NUMERIC  = ["price","minimum_nights","number_of_reviews",
            "reviews_per_month","availability_365"]
CATEG    = ["room_type","neighbourhood_group"]
FIG_DIR  = Path(__file__).parent / "figures"

# helper to build value-counts DF with reliable column names
def vc(series: pd.Series, new_name: str) -> pd.DataFrame:
    df = series.value_counts().reset_index()
    # pandas <2.0 uses 'index', >=2.0 uses series name
    if "index" in df.columns:
        df = df.rename(columns={"index": new_name, series.name: "Count"})
    else:
        df = df.rename(columns={series.name: new_name, "count": "Count"})
    return df

#  Dash app  #
app = Dash(__name__)
app.title = "Airbnb NYC Dashboard"

TITLE_STYLE = {"font": {"family": "Serif", "color": "blue", "size": 24}}
AXIS_STYLE  = {"title_font": {"family": "Serif", "color": "darkred", "size": 18}}

app.layout = html.Div([
    html.H1("Airbnb NYC 2019 Dashboard", style={"textAlign":"center"}),
    dcc.Store(id="df-store", data=df_clean.to_json(date_format="iso", orient="split")),
    dcc.Tabs(id="tabs", value="overview", children=[
        dcc.Tab(label="Overview",          value="overview"),
        dcc.Tab(label="Numeric Plots",     value="num"),
        dcc.Tab(label="Categorical Plots", value="cat"),
        dcc.Tab(label="Outliers",          value="out"),
        dcc.Tab(label="Transformation",    value="trans"),
        dcc.Tab(label="Normality",         value="norm"),
        dcc.Tab(label="PCA",               value="pca"),
        dcc.Tab(label="Statistics",        value="stats"),
        dcc.Tab(label="Static EDA",        value="static"),
    ]),
    html.Div(id="tab-content", style={"padding":"20px"})
])

#  Tab renderer  #
@app.callback(Output("tab-content","children"), Input("tabs","value"))
def render_tab(tab):
    if tab == "overview":
        # value-counts bar chart (unchanged)
        data = vc(df_clean["neighbourhood_group"], "Borough")
        bar = px.bar(data, x="Borough", y="Count", title="Listings per Borough")
        bar.update_layout(title=TITLE_STYLE)
        bar.update_xaxes(**AXIS_STYLE);
        bar.update_yaxes(**AXIS_STYLE)

        return html.Div([
            html.H3(f"Cleaned dataset — {len(df_clean):,} listings"),
            dash_table.DataTable(
                data=df_clean.round(2).to_dict("records"),
                columns=[{"name": c, "id": c} for c in df_clean.columns],
                page_current=0,
                page_size=20,  # 20 rows per page
                page_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"fontFamily": "Serif"}
            ),
            dcc.Graph(figure=bar)
        ])

    if tab=="num":
        return html.Div([
            html.Div([
                html.Label("X (numeric)"), dcc.Dropdown(NUMERIC,"price",id="num-x"),
                html.Br(),
                html.Label("Y (numeric)"), dcc.Dropdown(NUMERIC,"number_of_reviews",id="num-y")
            ],style={"width":"40%","display":"inline-block","verticalAlign":"top"}),
            html.Div([
                html.Label("Plot type"),
                dcc.RadioItems([{"label":t,"value":t} for t in ["Scatter","Line","Hexbin"]],
                               "Scatter", id="num-ptype", inline=True)
            ]),
            dcc.Graph(id="num-graph")
        ])

    if tab=="cat":
        return html.Div([
            html.Label("Categorical feature"), dcc.Dropdown(CATEG,"room_type",id="cat-col"),
            html.Br(),
            dcc.RadioItems([{"label":"Count","value":"count"},
                            {"label":"Average Price","value":"avg"}],
                           "count",id="cat-measure",inline=True),
            dcc.Graph(id="cat-graph")
        ])

    if tab=="out":
        return html.Div([
            html.Label("Numeric feature"), dcc.Dropdown(NUMERIC,"price",id="out-col"),
            dcc.Graph(id="out-box"), dcc.Graph(id="out-hist")
        ])

    if tab=="trans":
        return html.Div([
            html.Label("Numeric feature"), dcc.Dropdown(NUMERIC,"price",id="trans-col"),
            html.Br(),
            html.Label("Transformation"),
            dcc.RadioItems([{"label":"None","value":"none"},
                            {"label":"Log(x+1)","value":"log"},
                            {"label":"Square root","value":"sqrt"}],
                           "none",id="trans-type",inline=True),
            dcc.Graph(id="trans-graph")
        ])

    if tab=="norm":
        return html.Div([
            html.Label("Numeric feature"), dcc.Dropdown(NUMERIC,"price",id="norm-col"),
            html.Br(),
            dcc.RadioItems([{"label":"Histogram","value":"hist"},
                            {"label":"Q-Q plot","value":"qq"},
                            {"label":"Shapiro-Wilk","value":"shapiro"}],
                           "hist",id="norm-type",inline=True),
            html.Div(id="norm-out")
        ])

    if tab=="pca":
        return html.Div([
            html.Label("Numeric features"), dcc.Checklist(NUMERIC,["price","number_of_reviews"],
                                                          id="pca-cols",inline=True),
            html.Br(),
            dcc.RadioItems([{"label":"PC1 vs PC2","value":2},
                            {"label":"3-D","value":3}],
                           2,id="pca-dim",inline=True),
            dcc.Graph(id="pca-graph")
        ])

    if tab=="stats":
        stats = df_clean.describe().round(2).transpose().reset_index()
        return dash_table.DataTable(
            data=stats.to_dict("records"),
            columns=[{"name":c,"id":c} for c in stats.columns],
            style_table={"overflowX":"auto"}, style_cell={"fontFamily":"Serif"}
        )

    if tab=="static":
        gallery=[]
        if FIG_DIR.exists():
            captions={}
            rfile=FIG_DIR/"README.md"
            if rfile.exists():
                for ln in rfile.read_text(encoding="utf-8").splitlines():
                    if ln.startswith("- **"): fname,cap=ln[4:].split("** – "); captions[fname.strip()]=cap
            for png in sorted(FIG_DIR.glob("*.png")):
                img64=base64.b64encode(png.read_bytes()).decode()
                gallery.append(html.Div([
                    html.Img(src=f"data:image/png;base64,{img64}",
                             style={"height":"260px","border":"1px solid #ccc"}),
                    html.Br(),
                    html.Div(captions.get(png.name,""),style={"fontStyle":"italic","width":"260px"})
                ],style={"display":"inline-block","margin":"10px","textAlign":"center"}))
        return html.Div(gallery or "Run phase1_static_plots.py to generate figures.",
                        style={"padding":"10px"})

    return html.Div("Tab not found.")

#  Callbacks – numeric & categorical  #
def load_df(json_str): return pd.read_json(StringIO(json_str), orient="split")

@app.callback(Output("num-graph","figure"),
              [Input("num-x","value"), Input("num-y","value"), Input("num-ptype","value")],
              State("df-store","data"))
def num_plot(x,y,ptype,json_df):
    if not x or not y: raise PreventUpdate
    df=load_df(json_df)
    if ptype=="Scatter":
        fig=px.scatter(df,x=x,y=y,opacity=.6,title=f"{x} vs {y}")
    elif ptype=="Line":
        fig=px.line(df.sort_values(x),x=x,y=y,title=f"{y} over {x}")
    else:
        fig=px.density_heatmap(df,x=x,y=y,nbinsx=25,nbinsy=25,
                               title=f"{x} vs {y} Hexbin")
    fig.update_layout(title=TITLE_STYLE); fig.update_xaxes(**AXIS_STYLE); fig.update_yaxes(**AXIS_STYLE)
    return fig

@app.callback(Output("cat-graph","figure"),
              [Input("cat-col","value"), Input("cat-measure","value")],
              State("df-store","data"))
def cat_plot(col,measure,json_df):
    df=load_df(json_df)
    if measure=="count":
        data=vc(df[col],col)
        fig=px.bar(data,x=col,y="Count",title=f"Count by {col}")
    else:
        data=df.groupby(col)["price"].mean().reset_index()
        fig=px.bar(data,x=col,y="price",title=f"Average Price by {col}",
                   labels={"price":"Avg Price"})
    fig.update_layout(title=TITLE_STYLE); fig.update_xaxes(**AXIS_STYLE); fig.update_yaxes(**AXIS_STYLE)
    return fig

# Callbacks – outlier, transformation  #
@app.callback([Output("out-box","figure"), Output("out-hist","figure")],
              Input("out-col","value"), State("df-store","data"))
def outlier(col,json_df):
    df=load_df(json_df)
    box=px.box(df,y=col,title=f"{col} Box").update_layout(title=TITLE_STYLE)
    hist=px.histogram(df,x=col,nbins=30,title=f"{col} Histogram").update_layout(title=TITLE_STYLE)
    for fig in (box,hist):
        fig.update_xaxes(**AXIS_STYLE); fig.update_yaxes(**AXIS_STYLE)
    return box,hist

@app.callback(Output("trans-graph","figure"),
              [Input("trans-col","value"), Input("trans-type","value")],
              State("df-store","data"))
def transform(col,method,json_df):
    df=load_df(json_df); s=df[col]
    if method=="log": s=np.log1p(s); title=f"log(1+{col})"
    elif method=="sqrt": s=np.sqrt(s); title=f"√{col}"
    else: title=col
    fig=px.histogram(s,nbins=30,title=f"{title} Histogram")
    fig.update_layout(title=TITLE_STYLE); fig.update_xaxes(**AXIS_STYLE); fig.update_yaxes(**AXIS_STYLE)
    return fig

# Callbacks – normality & PCA #
@app.callback(Output("norm-out","children"),
              [Input("norm-col","value"), Input("norm-type","value")],
              State("df-store","data"))
def normality(col,method,json_df):
    df=load_df(json_df); data=df[col].dropna()
    if method=="hist":
        fig=px.histogram(data,nbins=30,title=f"{col} Histogram").update_layout(title=TITLE_STYLE)
        fig.update_xaxes(**AXIS_STYLE); fig.update_yaxes(**AXIS_STYLE)
        return dcc.Graph(figure=fig)
    if method=="qq":
        osm,osr=stats.probplot(data,dist="norm",plot=None)[:2]
        fig=go.Figure([go.Scatter(x=osr,y=osm,mode="markers"),
                       go.Scatter(x=osr,y=osr,mode="lines",line=dict(color="red"))])
        fig.update_layout(title={"text":f"{col} Q-Q Plot",**TITLE_STYLE["font"]},
                          xaxis_title="Theoretical",yaxis_title="Sample")
        fig.update_xaxes(**AXIS_STYLE); fig.update_yaxes(**AXIS_STYLE)
        return dcc.Graph(figure=fig)
    stat,p=stats.shapiro(data.sample(min(len(data),5000)))
    return html.Div(f"Shapiro-Wilk p-value: {p:.4f}",
                    style={"fontFamily":"Serif","fontSize":"18px"})

@app.callback(Output("pca-graph","figure"),
              [Input("pca-cols","value"), Input("pca-dim","value")],
              State("df-store","data"))
def pca_plot(cols,dim,json_df):
    if not cols or len(cols)<2: raise PreventUpdate
    df=load_df(json_df); X=df[cols].dropna()
    comps=PCA(n_components=dim).fit_transform(X)
    if dim==2:
        fig=px.scatter(x=comps[:,0],y=comps[:,1],
                       color=df.loc[X.index,"room_type"],
                       labels={"x":"PC1","y":"PC2"},title="PCA PC1 vs PC2")
    else:
        fig=px.scatter_3d(x=comps[:,0],y=comps[:,1],z=comps[:,2],
                          color=df.loc[X.index,"room_type"],
                          labels={"x":"PC1","y":"PC2","z":"PC3"},title="PCA 3-D")
    fig.update_layout(title=TITLE_STYLE)
    return fig

#  Main  #
if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--host",default="127.0.0.1")
    ap.add_argument("--port",type=int,default=8050)
    args=ap.parse_args()
    app.run(host=args.host, port=args.port, debug=False)