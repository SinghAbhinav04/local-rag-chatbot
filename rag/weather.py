"""
Weather fetcher — uses OpenWeather Geocoding API to get coordinates
for a city, then calls One Call 3.0 API to get an AI-generated
weather overview. Embeds the result directly into the active
vector DB session (no disk storage).
"""

import requests

from rag.config   import WEATHER_API_KEY
from rag.console  import console
from rag.chunking import chunk_text
from rag.vectordb import get_embedding


def get_lat_lon(city: str) -> tuple[float, float, str]:
    """Convert a city name to (lat, lon, official_name) via Geocoding API."""
    url = "http://api.openweathermap.org/geo/1.0/direct"
    resp = requests.get(url, params={
        "q": city,
        "limit": 1,
        "appid": WEATHER_API_KEY
    }, timeout=15)
    resp.raise_for_status()
    
    data = resp.json()
    if not data:
        raise ValueError(f"Could not find coordinates for '{city}'")
        
    loc = data[0]
    # Optionally include state/country for clarity if present
    name_parts = [loc["name"]]
    if "state" in loc: name_parts.append(loc["state"])
    if "country" in loc: name_parts.append(loc["country"])
    
    return loc["lat"], loc["lon"], ", ".join(name_parts)


def fetch_weather(city: str) -> tuple[str, str]:
    """Fetch weather overview for a city. Returns (official_name, overview_text)."""
    console.print(f"  [system]Geocoding '{city}'…[/]")
    lat, lon, official_name = get_lat_lon(city)
    
    console.print(f"  [system]Fetching weather for {official_name}…[/]")
    # Using the free Current Weather Data API instead of OneCall 3.0
    url = "https://api.openweathermap.org/data/2.5/weather"
    resp = requests.get(url, params={
        "lat": lat,
        "lon": lon,
        "appid": WEATHER_API_KEY,
        "units": "metric"
    }, timeout=15)
    
    resp.raise_for_status()
    data = resp.json()
    
    # Manually construct a weather overview string for the RAG from the JSON data
    desc = data.get("weather", [{}])[0].get("description", "unknown conditions")
    temp = data.get("main", {}).get("temp", "N/A")
    feels_like = data.get("main", {}).get("feels_like", "N/A")
    humidity = data.get("main", {}).get("humidity", "N/A")
    wind_speed = data.get("wind", {}).get("speed", "N/A")
    
    overview = (
        f"Right now in {official_name}, the weather is characterized by {desc}. "
        f"The temperature is {temp}°C, but it feels like {feels_like}°C. "
        f"The current humidity level is {humidity}%, and wind speeds are around {wind_speed} meters per second."
    )
        
    return official_name, overview


def weather_to_rag(collection, overview: str, location_name: str,
                   doc_chunk_counts: dict[str, int], chunk_offset: int) -> int:
    """Embed the weather overview directly into memory."""
    source_label = f"Weather: {location_name}"
    
    # We prefix it slightly so the model has clear context of what this text is
    full_text = f"Current Weather Overview for {location_name}:\n{overview}"
    
    chunks = chunk_text(full_text, source_label)
    if not chunks:
        console.print("  [info]No text to embed.[/]")
        return chunk_offset

    console.print(f"  [system]Embedding weather data into session memory…[/]")

    ids, embeddings, documents, metadatas = [], [], [], []
    for i, chunk in enumerate(chunks):
        cid = f"chunk_{chunk_offset + i}"
        emb = get_embedding(chunk["text"])
        ids.append(cid)
        embeddings.append(emb)
        documents.append(chunk["text"])
        metadatas.append({"source": source_label})

    collection.add(ids=ids, embeddings=embeddings,
                   documents=documents, metadatas=metadatas)
                   
    doc_chunk_counts[source_label] = len(chunks)
    console.print(f"  [system]✓ Weather data indexed (in-memory only).[/]")
    
    return chunk_offset + len(chunks)
