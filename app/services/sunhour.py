from geopy.geocoders import Nominatim

def get_peak_sun_hours(location: str):
    # In actual implementation, query Solcast or NREL API
    dummy_values = {"lagos": 4.5, "nairobi": 5.2, "cairo": 6.1}
    return dummy_values.get(location.lower(), 4.0)
