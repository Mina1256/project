import pandas as pd

# Your flights file now has a header row like:
# from_airport_code,from_country,dest_airport_code,dest_country,aircraft_type,
# airline_number,airline_name,flight_number,departure_time,arrival_time,
# duration,stops,price,currency,co2_emissions,avg_co2_emission_for_this_route,
# co2_percentage,scan_date

# ----------------------------
# 1) Load flights
# ----------------------------
def get_flights(flights_csv_path: str, *, max_stops: int = 0) -> pd.DataFrame:
    """
    Load flights CSV and return a cleaned DataFrame.

    - Uses the CSV header.
    - Keeps rows with stops <= max_stops (default 0).
    - Standardizes key columns to:
        origin, destination, stops, duration_min, price, currency,
        departure_time, arrival_time, co2_emissions
    Notes:
      - co2_emissions is kept as numeric. Many datasets store grams; you can
        convert later for display if you want (e.g., /1000 for kg).
    """
    df = pd.read_csv(flights_csv_path)

    # Trim whitespace in column names (defensive)
    df.columns = [c.strip() for c in df.columns]

    # Map your dataset columns -> our internal standard column names
    required_map = {
        "from_airport_code": "origin",
        "dest_airport_code": "destination",
        "stops": "stops",
        "duration": "duration_min",
        "price": "price",
        "currency": "currency",
        "departure_time": "departure_time",
        "arrival_time": "arrival_time",
        "co2_emissions": "co2_emissions",
    }

    # Rename to standard names (keep other columns too if you want)
    df = df.rename(columns=required_map)

    # Clean origin/destination
    df["origin"] = df["origin"].astype(str).str.strip().replace("", pd.NA)
    df["destination"] = df["destination"].astype(str).str.strip().replace("", pd.NA)
    df["currency"] = df["currency"].astype(str).str.strip().replace("", pd.NA)
    df["departure_time"] = df["departure_time"].astype(str).str.strip().replace("", pd.NA)
    df["arrival_time"] = df["arrival_time"].astype(str).str.strip().replace("", pd.NA)

    # Convert numeric columns
    df["stops"] = pd.to_numeric(df["stops"], errors="coerce")
    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["co2_emissions"] = pd.to_numeric(df["co2_emissions"], errors="coerce")

    # Drop rows with missing essentials
    df = df.dropna(subset=["origin", "destination", "stops", "duration_min", "price", "currency",
                           "departure_time", "arrival_time", "co2_emissions"])

    df["stops"] = df["stops"].astype(int)
    df["duration_min"] = df["duration_min"].astype(int)
    df["price"] = df["price"].astype(float)
    df["co2_emissions"] = df["co2_emissions"].astype(int)

    # Filter by stops
    df = df[df["stops"] <= int(max_stops)].copy()

    # Cast stops to int (clean)
    df["stops"] = df["stops"].astype(int)

    return df


# ----------------------------
# 2) Load airports
# ----------------------------
def get_airports(airports_csv_path: str) -> pd.DataFrame:
    """
    Load airports CSV with columns like:
    code,icao,name,latitude,longitude,...

    Returns a DataFrame with at least:
      - code (IATA)
      - latitude (float)
      - longitude (float)
      - name (optional)
    """
    airports = pd.read_csv(airports_csv_path)

    # Normalize code and coordinates
    airports["code"] = airports["code"].astype(str).str.strip()
    airports["latitude"] = pd.to_numeric(airports["latitude"], errors="coerce")
    airports["longitude"] = pd.to_numeric(airports["longitude"], errors="coerce")

    airports = airports.dropna(subset=["code", "latitude", "longitude"])

    return airports


# ----------------------------
# 3) Filter to "usable" airports + flights
# ----------------------------
def filter_to_connected_and_located(
    flights: pd.DataFrame,
    airports: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep:
      - Only flights where BOTH endpoints exist in airports with valid lat/lon
      - Only airports that appear in at least one remaining flight (as origin or destination)

    Returns:
      (filtered_flights, filtered_airports)
    """

    # Fast membership sets
    valid_codes = set(airports["code"].tolist())

    # Keep flights whose start and end are defined (present in airports_pos)
    flights_f = flights[
        flights["origin"].isin(valid_codes) & flights["destination"].isin(valid_codes)
    ].copy()

    # Keep only airports that have at least one flight to/from them
    used_codes = set(flights_f["origin"]).union(set(flights_f["destination"]))
    airports_f = airports[airports["code"].isin(used_codes)].copy()

    return flights_f, airports_f


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    flights = get_flights("flight data.csv", max_stops=2)
    airports = get_airports("airports.csv")

    flights_filtered, airports_filtered = filter_to_connected_and_located(flights, airports)

    print("Flights total:", len(flights), "-> filtered:", len(flights_filtered))
    print("Airports total:", len(airports), "-> filtered:", len(airports_filtered))
    print("Columns in flights:", list(flights_filtered.columns))
