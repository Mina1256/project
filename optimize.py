# optimize.py
from __future__ import annotations

import heapq
import math
from datetime import datetime, timedelta
from typing import Callable, Optional, Tuple, List

from graph import Graph, Flight

# ----------------------------
# Time parsing (match graph.py)
# ----------------------------
_TIME_FMT = "%Y-%m-%d %H:%M:%S"


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return datetime.strptime(s, _TIME_FMT)
    except ValueError:
        return None


# ----------------------------
# Helpers
# ----------------------------
def _safe_nonneg_float(x) -> Optional[float]:
    """Convert to a finite, nonnegative float. Return None if missing/invalid."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v) or v < 0:
        return None
    return v


# ----------------------------
# Constrained Dijkstra
# (same constraints as "no optimization" DFS)
# ----------------------------
def dijkstra_best_route_constrained(
    graph: Graph,
    start: str,
    dest: str,
    weight_fn: Callable[[Flight], Optional[float]],  # Passing a function
    *,
    k_legs: int,
    min_conn_hours: float,
    max_conn_hours: float = 24.0,
) -> Optional[Tuple[List[str], List[Flight], float]]:
    """
    Shortest path that respects the SAME feasibility rules as your DFS:

    - Total legs <= k_legs, where legs per Flight = stops + 1
    - Times parse, and dep <= arr
    - Connection time window:
         min_conn_hours <= next_dep - prev_arr <= max_conn_hours

    Returns:
        (airport_path, flight_path, total_cost) OR None if no feasible route exists.

    Notes:
    - weight_fn(flight) must return NONNEGATIVE weight.
    - Return None from weight_fn to skip a flight (e.g., missing price).
    """
    graph.get_airport(start)
    graph.get_airport(dest)

    min_gap = timedelta(hours=float(min_conn_hours))
    max_gap = timedelta(hours=float(max_conn_hours))

    # State includes last arrival time + legs used, because feasibility depends on them.
    State = tuple[str, int, Optional[datetime]]  # (airport, legs_used, last_arrival)

    start_state: State = (start, 0, None)

    dist: dict[State, float] = {start_state: 0.0} # Stores "Distance" at each state
    prev: dict[State, tuple[State, Flight]] = {}
    pq: list[tuple[float, str, int, Optional[datetime]]] = [(0.0, start, 0, None)] # Priority queue

    # Pruning per (airport, legs_used): keep a small set of nondominated labels (cost, arrival)
    # A dominates B if costA <= costB and arrivalA <= arrivalB (and at least one strict).
    frontier: dict[tuple[str, int], list[tuple[float, Optional[datetime]]]] = {(start, 0): [(0.0, None)]}

    def is_dominated(airport: str, legs: int, cost: float, arr: Optional[datetime]) -> bool:
        for c2, a2 in frontier.get((airport, legs), []):
            if a2 is None or arr is None:
                continue
            # An earlier arrival, gives more options for a next flight
            if c2 <= cost and a2 <= arr:
                return True
        return False

    def add_to_frontier(airport: str, legs: int, cost: float, arr: Optional[datetime]) -> None:
        lst = frontier.setdefault((airport, legs), [])
        new_lst: list[tuple[float, Optional[datetime]]] = []
        for c2, a2 in lst:
            if a2 is None or arr is None:
                new_lst.append((c2, a2))
                continue
            # If new dominates old, drop old
            if cost <= c2 and arr <= a2:
                continue
            new_lst.append((c2, a2))
        new_lst.append((cost, arr))
        frontier[(airport, legs)] = new_lst

    best_dest_state: Optional[State] = None
    best_dest_cost = float("inf")

    while pq:
        cost, u, legs_used, last_arrival = heapq.heappop(pq)
        state: State = (u, legs_used, last_arrival)

        if dist.get(state, float("inf")) != cost:
            continue  # stale

        if cost >= best_dest_cost:
            continue

        if u == dest:
            best_dest_cost = cost
            best_dest_state = state
            break  # Dijkstra property holds (nonnegative weights)

        for f in graph.flights_from(u):
            w = weight_fn(f)
            if w is None:
                continue
            if w < 0:
                raise ValueError("Dijkstra requires nonnegative weights.")

            # legs constraint
            new_legs = legs_used + 1
            if new_legs > k_legs:
                continue

            # time feasibility
            dep = _parse_dt(getattr(f, "departure_time", None))
            arr = _parse_dt(getattr(f, "arrival_time", None))
            if dep is None or arr is None:
                continue
            if dep > arr:
                continue

            # connection window
            if last_arrival is not None:
                gap = dep - last_arrival
                if gap < timedelta(0):
                    continue
                if gap < min_gap or gap > max_gap:
                    continue

            v = f.destination
            new_cost = cost + w
            new_state: State = (v, new_legs, arr)

            # prune dominated labels
            if is_dominated(v, new_legs, new_cost, arr):
                continue

            # Cost, then legs, then arrival time is compared
            if new_cost < dist.get(new_state, float("inf")):
                dist[new_state] = new_cost
                prev[new_state] = (state, f)
                add_to_frontier(v, new_legs, new_cost, arr)
                heapq.heappush(pq, (new_cost, v, new_legs, arr))

    if best_dest_state is None:
        return None

    # reconstruct
    airport_path: List[str] = []
    flight_path: List[Flight] = []
    cur = best_dest_state
    while cur != start_state:
        parent, used_flight = prev[cur]
        flight_path.append(used_flight)
        airport_path.append(cur[0])
        cur = parent
    airport_path.append(start)
    airport_path.reverse()
    flight_path.reverse()

    return airport_path, flight_path, best_dest_cost


# ----------------------------
# Metric-specific wrappers
# (NOW constrained + uses real emissions field)
# ----------------------------
def cheapest_route(
    graph: Graph,
    start: str,
    dest: str,
    *,
    k_legs: int,
    min_conn_hours: float,
    max_conn_hours: float = 24.0,
) -> Optional[Tuple[List[str], List[Flight], float]]:
    """Minimize total price, respecting legs + connection-time constraints."""
    def w(f: Flight) -> Optional[float]:
        return _safe_nonneg_float(getattr(f, "price", None))

    return dijkstra_best_route_constrained(
        graph, start, dest, w,
        k_legs=k_legs, min_conn_hours=min_conn_hours, max_conn_hours=max_conn_hours
    )


def fastest_route(
    graph: Graph,
    start: str,
    dest: str,
    *,
    k_legs: int,
    min_conn_hours: float,
    max_conn_hours: float = 24.0,
) -> Optional[Tuple[List[str], List[Flight], float]]:
    """Minimize total duration (minutes), respecting legs + connection-time constraints."""
    def w(f: Flight) -> Optional[float]:
        return _safe_nonneg_float(getattr(f, "duration_min", None))

    return dijkstra_best_route_constrained(
        graph, start, dest, w,
        k_legs=k_legs, min_conn_hours=min_conn_hours, max_conn_hours=max_conn_hours
    )


def lowest_emissions_route(
    graph: Graph,
    start: str,
    dest: str,
    *,
    k_legs: int,
    min_conn_hours: float,
    max_conn_hours: float = 24.0,
) -> Optional[Tuple[List[str], List[Flight], float]]:
    """
    Minimize total CO2 emissions using the ACTUAL per-flight emissions in your CSV:
        Flight.emissions

    (Units depend on your dataset; commonly grams. It's fine as long as you're consistent.)
    """
    def w(f: Flight) -> Optional[float]:
        return _safe_nonneg_float(getattr(f, "emissions", None))

    return dijkstra_best_route_constrained(
        graph, start, dest, w,
        k_legs=k_legs, min_conn_hours=min_conn_hours, max_conn_hours=max_conn_hours
    )


# ----------------------------
# Optional: pretty-printer
# ----------------------------
def describe_route(result: Optional[Tuple[List[str], List[Flight], float]], metric_name: str) -> None:
    if result is None:
        print(f"No route found for metric={metric_name}")
        return

    ap, fp, cost = result
    print(f"{metric_name} cost: {cost}")
    print("Airports:", " -> ".join(ap))
    for i, f in enumerate(fp, start=1):
        print(
            f"  {i}) {f.origin} -> {f.destination} | stops={f.stops} "
            f"| dep={f.departure_time} | arr={f.arrival_time} "
            f"| dur={f.duration_min} | price={f.price} {f.currency} "
            f"| emissions={f.emissions}"
        )
