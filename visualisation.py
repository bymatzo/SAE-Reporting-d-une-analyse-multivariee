"""
Visualisation — SAE 4.02
Graphiques Plotly pour la pré-analyse, l'ACP, l'ACM, l'AFC et la classification.
Toutes les fonctions prennent en entrée les dicts retournés par analysis.py.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram


# Palette commune
PALETTE = px.colors.qualitative.Set2


# ---------------------------------------------------------------------------
# Pré-analyse
# ---------------------------------------------------------------------------

def plot_missing_heatmap(missing_summary: pd.DataFrame) -> go.Figure:
    """
    Heatmap du taux de NaN par colonne (entrée : résultat de diagnose_missing).
    """
    df = missing_summary.sort_values("pct_missing", ascending=True)
    fig = go.Figure(go.Bar(
        x=df["pct_missing"],
        y=df.index,
        orientation="h",
        marker_color=df["pct_missing"],
        marker_colorscale="Reds",
        text=df["pct_missing"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
    ))
    fig.update_layout(
        title="Taux de valeurs manquantes par colonne",
        xaxis_title="% NaN",
        yaxis_title="",
        height=max(400, len(df) * 18),
        margin=dict(l=200),
    )
    return fig


def plot_univariate(df: pd.DataFrame, col: str) -> go.Figure:
    """
    Histogramme (variable quanti) ou barplot (variable quali).
    """
    if pd.api.types.is_numeric_dtype(df[col]):
        fig = px.histogram(
            df, x=col, nbins=40,
            title=f"Distribution — {col}",
            labels={col: col},
            color_discrete_sequence=[PALETTE[0]],
        )
        fig.update_traces(marker_line_width=0.5, marker_line_color="white")
    else:
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        fig = px.bar(
            counts, x=col, y="count",
            title=f"Fréquences — {col}",
            color_discrete_sequence=[PALETTE[1]],
        )
    fig.update_layout(bargap=0.05)
    return fig


def plot_correlation_matrix(df: pd.DataFrame, cols: list) -> go.Figure:
    """
    Heatmap de la matrice de corrélation de Pearson.
    """
    corr = df[cols].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        hoverongaps=False,
    ))
    fig.update_layout(
        title="Matrice de corrélation",
        height=500,
        width=600,
    )
    return fig


def plot_boxplot(df: pd.DataFrame, quanti_col: str, quali_col: str) -> go.Figure:
    """
    Boxplot d'une variable quantitative croisée avec une variable qualitative.
    """
    fig = px.box(
        df.dropna(subset=[quanti_col, quali_col]),
        x=quali_col, y=quanti_col,
        color=quali_col,
        title=f"{quanti_col} selon {quali_col}",
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# ACP
# ---------------------------------------------------------------------------

def plot_scree(explained_var: pd.Series) -> go.Figure:
    """
    Graphique des valeurs propres (éboulis) avec variance cumulée.
    """
    dims = explained_var.index.tolist()
    vals = explained_var.values
    cumul = np.cumsum(vals)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=dims, y=vals,
        name="% variance",
        marker_color=PALETTE[0],
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=dims, y=cumul,
        name="% cumulé",
        mode="lines+markers",
        marker_color=PALETTE[2],
        line_dash="dot",
    ), secondary_y=True)
    fig.add_hline(y=80, line_dash="dash", line_color="grey",
                  annotation_text="80%", secondary_y=True)
    fig.update_layout(
        title="Éboulis des valeurs propres",
        xaxis_title="Dimension",
        legend=dict(orientation="h", y=-0.15),
    )
    fig.update_yaxes(title_text="% variance expliquée", secondary_y=False)
    fig.update_yaxes(title_text="% cumulé", secondary_y=True)
    return fig


def plot_correlation_circle(
    loadings: pd.DataFrame,
    dim_x: int = 1,
    dim_y: int = 2,
    cos2_var: pd.DataFrame = None,
    cos2_threshold: float = 0.0,
) -> go.Figure:
    """
    Cercle des corrélations (ACP).
    Les variables sous cos2_threshold sont grisées.
    """
    dx, dy = f"Dim {dim_x}", f"Dim {dim_y}"
    lx, ly = loadings[dx].values, loadings[dy].values
    names = loadings.index.tolist()

    # Qualité de représentation pour la couleur
    if cos2_var is not None:
        quality = (cos2_var[dx] + cos2_var[dy]).values
    else:
        quality = np.ones(len(names))

    colors = [
        f"rgba(31,119,180,{max(0.3, q)})" if q >= cos2_threshold else "rgba(180,180,180,0.4)"
        for q in quality
    ]

    fig = go.Figure()

    # Cercle unité
    theta = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode="lines", line=dict(color="grey", width=1, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))

    # Flèches + labels
    for i, name in enumerate(names):
        fig.add_annotation(
            ax=0, ay=0, x=lx[i], y=ly[i],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1,
            arrowwidth=2, arrowcolor=colors[i],
        )
        fig.add_trace(go.Scatter(
            x=[lx[i]], y=[ly[i]],
            mode="text",
            text=[name],
            textposition="top center",
            textfont=dict(size=11),
            showlegend=False,
            hovertemplate=f"<b>{name}</b><br>{dx}: {lx[i]:.3f}<br>{dy}: {ly[i]:.3f}<extra></extra>",
        ))

    pct = lambda d: f"{explained_var_val:.1f}%" if (explained_var_val := 0) == 1 else ""

    fig.update_layout(
        title=f"Cercle des corrélations ({dx} × {dy})",
        xaxis=dict(title=dx, range=[-1.15, 1.15], zeroline=True, zerolinecolor="grey"),
        yaxis=dict(title=dy, range=[-1.15, 1.15], zeroline=True, zerolinecolor="grey",
                   scaleanchor="x"),
        width=550, height=550,
        showlegend=False,
    )
    return fig


def plot_individuals(
    scores: pd.DataFrame,
    dim_x: int = 1,
    dim_y: int = 2,
    color_col: pd.Series = None,
    labels_col: pd.Series = None,
    explained_var: pd.Series = None,
    alpha: float = 0.5,
) -> go.Figure:
    """
    Graphe des individus (ACP/ACM).
    color_col : Series de même index que scores pour colorier les points.
    labels_col : Series de même index pour afficher des étiquettes au survol.
    """
    dx, dy = f"Dim {dim_x}", f"Dim {dim_y}"
    sx = scores[dx].values
    sy = scores[dy].values

    x_title = f"{dx} ({explained_var[dx]:.1f}%)" if explained_var is not None else dx
    y_title = f"{dy} ({explained_var[dy]:.1f}%)" if explained_var is not None else dy

    hover = labels_col.values if labels_col is not None else [str(i) for i in scores.index]

    if color_col is not None:
        color_col = color_col.reindex(scores.index)
        categories = sorted(color_col.dropna().unique().tolist(), key=str)
        fig = go.Figure()
        for i, cat in enumerate(categories):
            mask = color_col == cat
            fig.add_trace(go.Scatter(
                x=sx[mask], y=sy[mask],
                mode="markers",
                name=str(cat),
                marker=dict(color=PALETTE[i % len(PALETTE)], opacity=alpha, size=5),
                hovertext=np.array(hover)[mask],
                hovertemplate="%{hovertext}<extra></extra>",
            ))
    else:
        fig = go.Figure(go.Scatter(
            x=sx, y=sy,
            mode="markers",
            marker=dict(color=PALETTE[0], opacity=alpha, size=4),
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False,
        ))

    fig.add_hline(y=0, line_color="grey", line_width=0.8)
    fig.add_vline(x=0, line_color="grey", line_width=0.8)
    fig.update_layout(
        title="Graphe des individus",
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title=color_col.name if color_col is not None else "",
        height=500,
    )
    return fig


def plot_cos2_bar(cos2: pd.DataFrame, dim: int = 1, top_n: int = 15) -> go.Figure:
    """
    Barplot des cos² pour une dimension donnée (variables ou individus).
    """
    d = f"Dim {dim}"
    data = cos2[d].sort_values(ascending=False).head(top_n)
    fig = go.Figure(go.Bar(
        x=data.index.astype(str),
        y=data.values,
        marker_color=PALETTE[3],
        text=[f"{v:.3f}" for v in data.values],
        textposition="outside",
    ))
    fig.add_hline(y=1 / len(cos2), line_dash="dash", line_color="red",
                  annotation_text="seuil moyen")
    fig.update_layout(
        title=f"Qualité de représentation (cos²) — {d}",
        xaxis_title="",
        yaxis_title="cos²",
        yaxis_range=[0, min(1.1, data.max() * 1.2)],
    )
    return fig


# ---------------------------------------------------------------------------
# ACM
# ---------------------------------------------------------------------------

def plot_mca_modalities(
    col_coords: pd.DataFrame,
    dim_x: int = 1,
    dim_y: int = 2,
    explained_var: pd.Series = None,
) -> go.Figure:
    """
    Graphe des modalités (ACM) coloré par variable.
    """
    dx, dy = f"Dim {dim_x}", f"Dim {dim_y}"
    x_title = f"{dx} ({explained_var[dx]:.1f}%)" if explained_var is not None else dx
    y_title = f"{dy} ({explained_var[dy]:.1f}%)" if explained_var is not None else dy

    # Récupère le nom de la variable à partir du nom de la modalité (format "var__modalité")
    def get_var(label):
        return label.split("__")[0] if "__" in label else "autre"

    col_coords = col_coords.copy()
    col_coords["_variable"] = [get_var(str(i)) for i in col_coords.index]
    col_coords["_label"] = [str(i).split("__")[-1] for i in col_coords.index]

    variables = sorted(col_coords["_variable"].unique())
    fig = go.Figure()
    for i, var in enumerate(variables):
        sub = col_coords[col_coords["_variable"] == var]
        fig.add_trace(go.Scatter(
            x=sub[dx], y=sub[dy],
            mode="markers+text",
            name=var,
            text=sub["_label"],
            textposition="top center",
            marker=dict(size=10, color=PALETTE[i % len(PALETTE)]),
            hovertemplate="<b>%{text}</b><br>" + dx + ": %{x:.3f}<br>" + dy + ": %{y:.3f}<extra></extra>",
        ))

    fig.add_hline(y=0, line_color="grey", line_width=0.8)
    fig.add_vline(x=0, line_color="grey", line_width=0.8)
    fig.update_layout(
        title="Graphe des modalités (ACM)",
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=550,
        legend_title="Variable",
    )
    return fig


# ---------------------------------------------------------------------------
# AFC
# ---------------------------------------------------------------------------

def plot_ca_biplot(
    row_coords: pd.DataFrame,
    col_coords: pd.DataFrame,
    dim_x: int = 1,
    dim_y: int = 2,
    explained_var: pd.Series = None,
    row_label: str = "lignes",
    col_label: str = "colonnes",
) -> go.Figure:
    """
    Biplot AFC : modalités lignes et colonnes sur le même plan.
    """
    dx, dy = f"Dim {dim_x}", f"Dim {dim_y}"
    x_title = f"{dx} ({explained_var[dx]:.1f}%)" if explained_var is not None else dx
    y_title = f"{dy} ({explained_var[dy]:.1f}%)" if explained_var is not None else dy

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=row_coords[dx], y=row_coords[dy],
        mode="markers+text",
        name=row_label,
        text=row_coords.index.astype(str),
        textposition="top center",
        marker=dict(symbol="circle", size=10, color=PALETTE[0]),
    ))
    fig.add_trace(go.Scatter(
        x=col_coords[dx], y=col_coords[dy],
        mode="markers+text",
        name=col_label,
        text=col_coords.index.astype(str),
        textposition="top center",
        marker=dict(symbol="diamond", size=10, color=PALETTE[1]),
    ))
    fig.add_hline(y=0, line_color="grey", line_width=0.8)
    fig.add_vline(x=0, line_color="grey", line_width=0.8)
    fig.update_layout(
        title="Biplot AFC",
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def plot_dendrogram(linkage_matrix: np.ndarray, n_last: int = 30) -> go.Figure:
    """
    Dendrogramme (CAH) — affiche les n_last dernières fusions.
    """
    ddata = dendrogram(linkage_matrix, truncate_mode="lastp", p=n_last, no_plot=True)

    fig = go.Figure()
    for xs, ys in zip(ddata["icoord"], ddata["dcoord"]):
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color=PALETTE[0], width=1.5),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        title=f"Dendrogramme (CAH — {n_last} dernières fusions)",
        xaxis=dict(showticklabels=False, title="Individus"),
        yaxis_title="Distance de fusion",
        height=400,
    )
    return fig


def plot_elbow(elbow_result: dict) -> go.Figure:
    """
    Courbe du coude : inertie et silhouette en fonction de k.
    """
    k_vals = elbow_result["k_values"]
    inertias = elbow_result["inertias"]
    silhouettes = elbow_result["silhouette_scores"]
    best_k = elbow_result["best_k_silhouette"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=k_vals, y=inertias,
        name="Inertie intra",
        mode="lines+markers",
        marker_color=PALETTE[0],
        line_width=2,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=k_vals, y=silhouettes,
        name="Silhouette",
        mode="lines+markers",
        marker_color=PALETTE[1],
        line_width=2,
        line_dash="dot",
    ), secondary_y=True)
    fig.add_vline(
        x=best_k, line_dash="dash", line_color="red",
        annotation_text=f"k={best_k}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Méthode du coude — K-Moyennes",
        xaxis=dict(title="Nombre de clusters k", tickmode="linear"),
        legend=dict(orientation="h", y=-0.15),
    )
    fig.update_yaxes(title_text="Inertie intra-clusters", secondary_y=False)
    fig.update_yaxes(title_text="Score de silhouette", secondary_y=True)
    return fig


def plot_cluster_profile(quanti_profile: pd.DataFrame) -> go.Figure:
    """
    Radar chart du profil moyen de chaque cluster (valeurs centrées-réduites).
    """
    # Centrage-réduction pour rendre les variables comparables
    profile_std = (quanti_profile - quanti_profile.mean()) / quanti_profile.std().replace(0, 1)
    cols = profile_std.columns.tolist()
    cols_closed = cols + [cols[0]]  # fermeture du polygone

    fig = go.Figure()
    for i, cluster in enumerate(profile_std.index):
        vals = profile_std.loc[cluster].tolist()
        vals_closed = vals + [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=cols_closed,
            fill="toself",
            name=str(cluster),
            line_color=PALETTE[i % len(PALETTE)],
            opacity=0.7,
        ))

    fig.update_layout(
        title="Profil des clusters (valeurs centrées-réduites)",
        polar=dict(radialaxis=dict(visible=True)),
        height=500,
        legend_title="Cluster",
    )
    return fig


def plot_clusters_on_pca(
    scores: pd.DataFrame,
    labels: pd.Series,
    dim_x: int = 1,
    dim_y: int = 2,
    explained_var: pd.Series = None,
) -> go.Figure:
    """
    Projette les clusters k-moyennes sur le plan factoriel de l'ACP.
    """
    return plot_individuals(
        scores,
        dim_x=dim_x,
        dim_y=dim_y,
        color_col=labels,
        explained_var=explained_var,
    )
