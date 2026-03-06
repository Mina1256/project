# app.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# project modules
from data import get_flights, get_airports, filter_to_connected_and_located
from graph import build_graph, find_itineraries_within_k_legs_time_pruned

# optimizer (Dijkstra wrappers)
from optimize import cheapest_route, fastest_route, lowest_emissions_route

# -----------------------------
# Color helper (distinct per path)
# -----------------------------
def path_color(idx: int) -> str:
    golden_angle = 137.508
    hue = (idx * golden_angle) % 360
    return f"hsl({hue:.1f}, 75%, 50%)"


# -----------------------------
# Plot helpers (MAP SAME AS FIRST FILE)
# -----------------------------
def add_airports_layer(fig: go.Figure, airports_df, start_code: str, dest_code: str):
    airports_df = airports_df.copy()
    airports_df["code"] = airports_df["code"].astype(str).str.upper()
    airports_df["label"] = airports_df["code"]

    normal_df = airports_df[
        (airports_df["code"] != start_code) & (airports_df["code"] != dest_code)
    ]

    fig.add_trace(
        go.Scattergeo(
            lat=normal_df["latitude"],
            lon=normal_df["longitude"],
            mode="markers+text",
            text=normal_df["label"],
            textposition="top center",
            hovertext=[
                f"{row['code']} — {row.get('name', '')}<br>({row['latitude']:.4f}, {row['longitude']:.4f})"
                for _, row in normal_df.iterrows()
            ],
            hoverinfo="text",
            marker=dict(size=6),
            name="Airports",
        )
    )

    start_df = airports_df[airports_df["code"] == start_code]
    if not start_df.empty:
        fig.add_trace(
            go.Scattergeo(
                lat=start_df["latitude"],
                lon=start_df["longitude"],
                mode="markers+text",
                text=[start_code],
                textposition="top center",
                hovertext=[
                    f"START: {row['code']} — {row.get('name', '')}<br>({row['latitude']:.4f}, {row['longitude']:.4f})"
                    for _, row in start_df.iterrows()
                ],
                hoverinfo="text",
                marker=dict(size=14, color="green", line=dict(width=1, color="black")),
                name="Start",
            )
        )

    dest_df = airports_df[airports_df["code"] == dest_code]
    if not dest_df.empty:
        fig.add_trace(
            go.Scattergeo(
                lat=dest_df["latitude"],
                lon=dest_df["longitude"],
                mode="markers+text",
                text=[dest_code],
                textposition="top center",
                hovertext=[
                    f"DEST: {row['code']} — {row.get('name', '')}<br>({row['latitude']:.4f}, {row['longitude']:.4f})"
                    for _, row in dest_df.iterrows()
                ],
                hoverinfo="text",
                marker=dict(size=14, color="red", line=dict(width=1, color="black")),
                name="Destination",
            )
        )


def add_itinerary_paths(fig: go.Figure, airports_df, itineraries, num_points: int = 150):
    """
    Draw each itinerary as a separate colored line.
    """
    lookup: dict[str, tuple[float, float, str]] = {}
    for _, row in airports_df.iterrows():
        lookup[str(row["code"]).upper()] = (
            float(row["latitude"]),
            float(row["longitude"]),
            str(row.get("name", "")),
        )

    for idx, (airport_path, flight_path) in enumerate(itineraries, start=1):
        coords = []
        ok = True
        for code in airport_path:
            code = str(code).upper()
            if code not in lookup:
                ok = False
                break
            coords.append(lookup[code])
        if not ok or len(coords) < 2:
            continue

        lats: List[Optional[float]] = []
        lons: List[Optional[float]] = []

        for i in range(len(coords) - 1):
            (lat1, lon1, _), (lat2, lon2, _) = coords[i], coords[i + 1]
            lats.extend(np.linspace(lat1, lat2, num_points))
            lons.extend(np.linspace(lon1, lon2, num_points))
            if i != len(coords) - 2:
                lats.append(None)
                lons.append(None)

        legs_lines = []
        for j, f in enumerate(flight_path, start=1):
            legs_lines.append(
                f"{j}) {f.origin} → {f.destination} | stops: {getattr(f,'stops',0)} | "
                f"dep: {getattr(f,'departure_time',None)} | arr: {getattr(f,'arrival_time',None)}"
            )
        legs_html = "<br>".join(legs_lines) if legs_lines else "No flight details"

        hover = (
            f"<b>Path {idx}</b><br>"
            f"{' → '.join(airport_path)}<br>"
            f"<br><b>Leg breakdown</b><br>{legs_html}"
        )

        fig.add_trace(
            go.Scattergeo(
                lat=lats,
                lon=lons,
                mode="lines",
                line=dict(width=3, color=path_color(idx)),
                hoverinfo="text",
                text=[hover] * len(lats),
                name=f"Path {idx}",
            )
        )


# -----------------------------
# Data loading
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
DEFAULT_FLIGHTS = APP_DIR / "flight data.csv"
DEFAULT_AIRPORTS = APP_DIR / "airports.csv"


@st.cache_data(show_spinner=False)
def load_airports_and_flights(flights_path: str, airports_path: str):
    flights = get_flights(flights_path)
    airports = get_airports(airports_path)
    flights_f, airports_f = filter_to_connected_and_located(flights, airports)

    airports_f = airports_f.copy()
    airports_f["code"] = airports_f["code"].astype(str).str.upper()
    return flights_f, airports_f


@st.cache_resource(show_spinner=False)
def load_graph(flights_path: str, airports_path: str):
    return build_graph(flights_path, airports_path)


# -----------------------------
# Details under map
# -----------------------------
def _fmt_money(price: Any, currency: Any) -> str:
    if price is None:
        return "N/A"
    try:
        return f"{float(price):.2f} {currency or ''}".strip()
    except Exception:
        return f"{price} {currency or ''}".strip()


def render_flight_details(path_idx: int, airport_path: List[str], flight_path: List[Any]) -> None:
    st.subheader(f"Flight details — Path {path_idx}")
    st.write("**Airports:**", " → ".join(airport_path))

    if not flight_path:
        st.info("No flight legs to show.")
        return

    for i, f in enumerate(flight_path, start=1):
        st.markdown(
            f"""
**{i}) {getattr(f, 'origin', '?')} → {getattr(f, 'destination', '?')}**  
- stops: `{getattr(f, 'stops', 0)}`
- dep: `{getattr(f, 'departure_time', None)}`
- arr: `{getattr(f, 'arrival_time', None)}`
- duration (min): `{getattr(f, 'duration_min', None)}`
- price: `{_fmt_money(getattr(f, 'price', None), getattr(f, 'currency', ''))}`
- CO₂ emissions: `{getattr(f, 'emissions', None)}`
""".strip()
        )


# -----------------------------
# Main app
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="Flight Optimizer (Graph + Dijkstra)", layout="wide")
    st.title("Flight Optimizer (Graph + Dijkstra)")

    with st.sidebar:
        flights_path = str(DEFAULT_FLIGHTS)
        airports_path = str(DEFAULT_AIRPORTS)

        st.header("Search constraints")
        max_stops = st.number_input("Max stops (connections)", min_value=0, max_value=10, value=2, step=1)
        k_legs = int(max_stops) + 1
        min_conn_hours = st.number_input("Min connection hours", min_value=0.0, max_value=24.0, value=1.0, step=0.5)
        max_conn_hours = st.number_input("Max connection hours", min_value=1.0, max_value=72.0, value=24.0, step=1.0)

        max_results = 50  # behind-the-scenes

    # Load data
    try:
        _, airports_f = load_airports_and_flights(flights_path, airports_path)
        graph = load_graph(flights_path, airports_path)
    except FileNotFoundError as e:
        st.error(f"FileNotFoundError: {e}")
        st.info("Put `flight data.csv` and `airports.csv` next to app.py.")
        return
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    # State init
    if "mode" not in st.session_state:
        st.session_state["mode"] = "none"
    if "selected_path_idx" not in st.session_state:
        st.session_state["selected_path_idx"] = None  # 1-based; None means "nothing selected"

    # Airport selection
    codes = sorted(set(airports_f["code"].astype(str).str.upper()))
    if len(codes) < 2:
        st.error("Not enough airports after filtering.")
        return

    colA, colB = st.columns(2)
    with colA:
        START = st.selectbox("Start airport", options=codes, index=0)
    with colB:
        DEST = st.selectbox("Destination airport", options=codes, index=1)

    if START == DEST:
        st.warning("Pick different start/destination airports.")
        return

    # Mode buttons (instant recolor + rerun)
    b1, b2, b3, b4 = st.columns(4)

    with b1:
        if st.button(
            "No optimization",
            type=("primary" if st.session_state["mode"] == "none" else "secondary"),
            use_container_width=True,
            key="mode_none",
        ):
            st.session_state["mode"] = "none"
            st.session_state["selected_path_idx"] = None
            st.rerun()

    with b2:
        if st.button(
            "Optimize: Cheapest",
            type=("primary" if st.session_state["mode"] == "cheapest" else "secondary"),
            use_container_width=True,
            key="mode_cheapest",
        ):
            st.session_state["mode"] = "cheapest"
            st.session_state["selected_path_idx"] = None
            st.rerun()

    with b3:
        if st.button(
            "Optimize: Fastest",
            type=("primary" if st.session_state["mode"] == "fastest" else "secondary"),
            use_container_width=True,
            key="mode_fastest",
        ):
            st.session_state["mode"] = "fastest"
            st.session_state["selected_path_idx"] = None
            st.rerun()

    with b4:
        if st.button(
            "Optimize: Lowest CO₂",
            type=("primary" if st.session_state["mode"] == "emissions" else "secondary"),
            use_container_width=True,
            key="mode_emissions",
        ):
            st.session_state["mode"] = "emissions"
            st.session_state["selected_path_idx"] = None
            st.rerun()

    mode = st.session_state["mode"]
    st.caption(
        f"Mode: **{mode}** | Stops ≤ {max_stops} (legs ≤ {k_legs}) | "
        f"Connection window: {min_conn_hours}h – {max_conn_hours}h"
    )

    # Compute routes
    itineraries_for_plot: List[Tuple[List[str], List]] = []
    metric_label = ""

    if mode == "none":
        with st.spinner("Searching feasible itineraries (DFS + pruning)..."):
            itineraries_for_plot = find_itineraries_within_k_legs_time_pruned(
                graph,
                start=START,
                dest=DEST,
                k=k_legs,
                min_conn_hours=float(min_conn_hours),
                max_conn_hours=float(max_conn_hours),
                max_results=int(max_results),
            )
        st.write(f"Found **{len(itineraries_for_plot)}** feasible itineraries.")
    else:
        with st.spinner("Optimizing (Dijkstra)..."):
            if mode == "cheapest":
                result = cheapest_route(
                    graph, START, DEST,
                    k_legs=k_legs,
                    min_conn_hours=float(min_conn_hours),
                    max_conn_hours=float(max_conn_hours),
                )
                metric_label = "Cheapest"
            elif mode == "fastest":
                result = fastest_route(
                    graph, START, DEST,
                    k_legs=k_legs,
                    min_conn_hours=float(min_conn_hours),
                    max_conn_hours=float(max_conn_hours),
                )
                metric_label = "Fastest"
            else:
                result = lowest_emissions_route(
                    graph, START, DEST,
                    k_legs=k_legs,
                    min_conn_hours=float(min_conn_hours),
                    max_conn_hours=float(max_conn_hours),
                )
                metric_label = "Lowest CO₂"

        if result is None:
            st.error(f"No {metric_label.lower()} route found from {START} to {DEST}.")
            itineraries_for_plot = []
        else:
            airport_path, flight_path, total_cost = result
            st.success(f"{metric_label} route found. Score = {total_cost:.2f}")
            itineraries_for_plot = [(airport_path, flight_path)]

    # If we have paths, let user choose which one to show details for (replaces clicking)
    if itineraries_for_plot:
        if st.session_state["selected_path_idx"] is None:
            st.session_state["selected_path_idx"] = 1

        st.session_state["selected_path_idx"] = int(
            st.selectbox(
                "Which path’s details do you want to show?",
                options=list(range(1, len(itineraries_for_plot) + 1)),
                index=int(st.session_state["selected_path_idx"]) - 1,
            )
        )

    # Build plot
    fig = go.Figure()
    if itineraries_for_plot:
        add_itinerary_paths(fig, airports_f, itineraries_for_plot, num_points=200)

    add_airports_layer(fig, airports_f, START, DEST)

    fig.update_geos(projection_type="orthographic", showland=True, showcountries=True)
    fig.update_layout(
        height=850,
        margin=dict(r=0, t=60, l=0, b=0),
        paper_bgcolor="white",
        font=dict(color="black"),
        title=f"Flight paths {START} → {DEST} (mode: {mode})",
        legend=dict(orientation="h"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Details under map
    if itineraries_for_plot:
        idx = int(st.session_state["selected_path_idx"])
        idx = max(1, min(idx, len(itineraries_for_plot)))
        ap, fp = itineraries_for_plot[idx - 1]
        render_flight_details(idx, ap, fp)
    else:
        st.info("No route to show details for.")


if __name__ == "__main__":
    main()
