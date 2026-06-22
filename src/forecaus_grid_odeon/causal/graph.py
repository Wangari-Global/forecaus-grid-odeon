"""Causal DAG for the energy system (the 'causal companion' to the physics twin).

This is an explicit, hypothesised structural model over columns that exist in the
modelling frame, so it can be handed straight to DoWhy. Edges (cause -> effect):

  calendar (hour, weekend, holiday) -> demand        # activity rhythm
  calendar                          -> temperature   # diurnal cycle (confounder)
  calendar                          -> irradiance    # daylight
  calendar                          -> price         # daily price shape
  temperature                       -> demand        # heating / cooling  (TREATMENT)
  temperature                       -> price         # weather-driven supply/demand
  irradiance, wind                  -> price         # renewable supply lowers price
  price                             -> demand        # price response

Note the calendar -> {temperature, demand} fork: time-of-day is a genuine
*confounder* of the temperature -> demand effect, which is exactly why a naive
correlation is biased and a backdoor adjustment is needed. The graph is acyclic
by construction (demand is a sink).
"""
from __future__ import annotations

from typing import Optional

import networkx as nx

# Default effect of interest for the TRL-4 demo.
TREATMENT = "temp_c"
OUTCOME = "load_mw"

# Calendar drivers present in the feature frame.
_CALENDAR = ["hour_sin", "hour_cos", "is_weekend", "is_holiday"]


def build_dag(treatment: str = TREATMENT, outcome: str = OUTCOME) -> nx.DiGraph:
    """Return the hypothesised causal DAG as a :class:`networkx.DiGraph`."""
    g = nx.DiGraph()
    price = "day_ahead_price_eur_mwh"

    # calendar -> demand, temperature, irradiance, price
    for c in _CALENDAR:
        g.add_edge(c, outcome)
        g.add_edge(c, price)
    g.add_edge("hour_sin", "temp_c")
    g.add_edge("hour_cos", "temp_c")
    g.add_edge("hour_sin", "irradiance_wm2")
    g.add_edge("hour_cos", "irradiance_wm2")

    # weather -> demand / price
    g.add_edge("temp_c", outcome)            # the effect we estimate
    g.add_edge("temp_c", price)
    g.add_edge("irradiance_wm2", price)
    g.add_edge("wind_ms", price)

    # price -> demand
    g.add_edge(price, outcome)

    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("constructed graph is not a DAG")
    # Make sure the requested effect is representable.
    g.add_node(treatment)
    g.add_node(outcome)
    return g


def to_dot(graph: nx.DiGraph) -> str:
    """Serialise to a DOT string DoWhy can parse."""
    edges = "\n".join(f'  "{u}" -> "{v}";' for u, v in graph.edges())
    nodes = "\n".join(f'  "{n}";' for n in graph.nodes())
    return f"digraph G {{\n{nodes}\n{edges}\n}}"


def required_columns(graph: nx.DiGraph) -> list[str]:
    """Columns the data must contain to fit this graph."""
    return list(graph.nodes())


def causal_parents(graph: nx.DiGraph, node: str) -> list[str]:
    """Direct causes (parents) of ``node`` — the causal feature set.

    Selecting the outcome's parents (and excluding its *effects*/children) is the
    intervention-aware feature selection used by the causal-augmented forecaster:
    a parent's effect on the outcome is invariant under interventions elsewhere,
    whereas an effect/child of the outcome can decouple under a regime change.
    """
    return list(graph.predecessors(node))


def draw_dag(graph: Optional[nx.DiGraph] = None, ax=None):
    """Draw the DAG (used by the causal-analysis notebook). Returns the Axes."""
    import matplotlib.pyplot as plt

    graph = graph if graph is not None else build_dag()
    pos = nx.spring_layout(graph, seed=42, k=1.2)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))
    nx.draw_networkx(
        graph, pos, ax=ax, with_labels=True, node_color="#cfe3f7",
        node_size=2400, font_size=8, arrowsize=16, edge_color="#888",
    )
    ax.set_title("forecaus-grid causal DAG — energy demand / price drivers")
    ax.axis("off")
    return ax
