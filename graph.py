from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional, List, Tuple

# Your loader module (you named it data.py)
from data import get_flights, get_airports, filter_to_connected_and_located


# ============================
# Data types
# ============================
@dataclass(frozen=True)
class Flight:
    origin: str
    destination: str
    stops: int
    emissions: float
    departure_time: Optional[str] = None  # "YYYY-MM-DD HH:MM:SS"
    arrival_time: Optional[str] = None    # "YYYY-MM-DD HH:MM:SS"
    duration_min: Optional[int] = None
    price: Optional[float] = None
    currency: Optional[str] = None


@dataclass
class Airport:
    code: str  # IATA
    name: str
    lat: float
    lon: float
    outgoing: List[Flight]  # list of flights leaving this airport

    def __init__(self, code: str, name: str, lat: float, lon: float) -> None:
        self.code = code
        self.name = name
        self.lat = lat
        self.lon = lon
        self.outgoing = []


# ============================
# Graph (CSC111-ish style)
# ============================
class _Vertex:
    item: Any  # Airport

    def __init__(self, item: Any) -> None:
        self.item = item


class Graph:
    """Directed graph:
    - nodes are airports
    - each Airport stores flights leaving it (Airport.outgoing)
    """
    _vertices: dict[str, _Vertex]

    def __init__(self) -> None:
        self._vertices = {}

    def add_airport(self, airport: Airport) -> None:
        if airport.code not in self._vertices:
            self._vertices[airport.code] = _Vertex(airport)

    def add_flight(self, flight: Flight) -> None:
        if flight.origin not in self._vertices or flight.destination not in self._vertices:
            raise ValueError("Both origin and destination airports must exist in the graph.")
        origin_airport: Airport = self._vertices[flight.origin].item
        origin_airport.outgoing.append(flight)

    def get_airport(self, code: str) -> Airport:
        if code not in self._vertices:
            raise KeyError(f"Airport {code} not found.")
        return self._vertices[code].item

    def airports(self) -> List[Airport]:
        return [v.item for v in self._vertices.values()]

    def flights_from(self, code: str) -> List[Flight]:
        return list(self.get_airport(code).outgoing)

    def __contains__(self, code: str) -> bool:
        return code in self._vertices

    def __len__(self) -> int:
        return len(self._vertices)


# ============================
# Build graph from CSVs
# ============================
def build_graph(flights_csv: str, airports_csv: str) -> Graph:
    flights_df = get_flights(flights_csv)
    airports_df = get_airports(airports_csv)

    flights_f, airports_f = filter_to_connected_and_located(flights_df, airports_df)

    graph = Graph()

    # Add airports
    for _, row in airports_f.iterrows():
        graph.add_airport(
            Airport(
                code=str(row["code"]).strip(),
                name=str(row.get("name", "")).strip(),
                lat=float(row["latitude"]),
                lon=float(row["longitude"]),
            )
        )

    # Add flights using named columns (NOT numeric indices)
    for _, row in flights_f.iterrows():
        origin = row["origin"]
        dest = row["destination"]

        stops = row["stops"]

        dep = row["departure_time"]
        arr = row["arrival_time"]

        duration = row["duration_min"]
        price = row["price"]
        currency = row["currency"]

        emissions = row["co2_emissions"]

        flight = Flight(
            origin=origin,
            destination=dest,
            stops=stops,
            emissions=emissions,
            departure_time=dep,
            arrival_time=arr,
            duration_min=duration,
            price=price,
            currency=currency,
        )

        graph.add_flight(flight)

    return graph


# ============================
# Fast itinerary search with time pruning
# Total legs = sum(stops + 1)
# ============================
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


def find_itineraries_within_k_legs_time_pruned(
    graph: Graph,
    start: str,
    dest: str,
    k: int,
    min_conn_hours: float,
    max_conn_hours: float = 24.0,
    max_results: int = 200,
) -> List[Tuple[List[str], List[Flight]]]:
    """
    Return up to max_results itineraries start -> dest such that:
      - total legs <= k, where legs per Flight = stops + 1
      - chronological, and each connection satisfies:
            min_conn_hours <= next_dep - prev_arr <= max_conn_hours
    Prunes while searching (fast).
    """
    graph.get_airport(start)
    graph.get_airport(dest)

    min_gap = timedelta(hours=min_conn_hours)
    max_gap = timedelta(hours=max_conn_hours)

    results: list[Tuple[list[str], list[Flight]]] = []

    def dfs(curr: str, legs_used: int, airport_path: list[str], flight_path: list[Flight],
            visited: set[str], last_arrival: Optional[datetime]) -> None:
        if len(results) >= max_results:
            return
        if legs_used > k:
            return
        if curr == dest:
            results.append((airport_path.copy(), flight_path.copy()))
            return

        for f in graph.flights_from(curr):
            if len(results) >= max_results:
                return

            nxt = f.destination
            if nxt in visited:
                continue

            new_legs = legs_used + 1
            if new_legs > k:
                continue

            dep = _parse_dt(f.departure_time)
            arr = _parse_dt(f.arrival_time)
            if dep is None or arr is None:
                continue
            if dep > arr:
                continue

            if last_arrival is not None:
                gap = dep - last_arrival
                if gap < timedelta(0):
                    continue
                if gap < min_gap or gap > max_gap:
                    continue

            visited.add(nxt)
            airport_path.append(nxt)
            flight_path.append(f)

            dfs(nxt, new_legs, airport_path, flight_path, visited, arr)

            flight_path.pop()
            airport_path.pop()
            visited.remove(nxt)

    dfs(start, 0, [start], [], {start}, None)
    return results


# ============================
# Demo / check
# ============================
if __name__ == "__main__":
    graph = build_graph("flight data.csv", "airports.csv")

    print("Airports in graph:", len(graph))
    total_edges = sum(len(a.outgoing) for a in graph.airports())
    print("Flights (edges) in graph:", total_edges)

    itins = find_itineraries_within_k_legs_time_pruned(
        graph, start="ALG", dest="AEP", k=2,
        min_conn_hours=1.0, max_conn_hours=24.0,
        max_results=5
    )
    print("Found itineraries:", len(itins))
