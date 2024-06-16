import folium
import overpy
import geojson
import pandas as pd
import json
import re
import requests
import time
from geopy.distance import geodesic

# Function to convert a string to a valid JSON format
def convert_to_dict(s):
    if isinstance(s, str):
        s = re.sub(r'(\w+)=', r'"\1":', s)
        s = s.replace('=', ':')
        s = s.replace('Point', '"Point"')
        return json.loads(s)
    return None

# Function to get the building outline using Overpass API
def get_building_contour(lat, lon):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      way(around:10,{lat},{lon})["building"];
      relation(around:10,{lat},{lon})["building"];
    );
    out body;
    >;
    out skel qt;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()

    building_coordinates = []
    for element in data['elements']:
        if element['type'] == 'way':
            nodes = element['nodes']
            for node in nodes:
                for el in data['elements']:
                    if el['id'] == node and 'lat' in el and 'lon' in el:
                        building_coordinates.append([el['lat'], el['lon']])
    return building_coordinates

# Read data from Excel file
file_path = 'cleaned_excel_file_5.xlsx'  # Specify your file path here
data = pd.read_excel(file_path)

# Group sub-buildings by the main building
main_buildings = {}
for index, row in data.iterrows():
    main_building_key = row['geodata_center']
    main_building_coordinates_dict = convert_to_dict(main_building_key)
    if main_building_coordinates_dict is None:
        continue
    main_building_coordinates = main_building_coordinates_dict['coordinates'][::-1]  # Reverse coordinates for folium
    if main_building_key not in main_buildings:
        main_buildings[main_building_key] = {
            'coordinates': main_building_coordinates,
            'sub_buildings': [],
            'main_building_border' : get_building_contour(*main_building_coordinates)
        }

    #main_buildings['main_building_border'] = get_building_contour(*main_buildings['coordinates'])
    latitude = row['Широта']
    longitude = row['Долгота']
    building_contour = get_building_contour(latitude, longitude)
    if building_contour:  # Skip empty contours
        distance = geodesic(main_building_coordinates, [latitude, longitude]).kilometers
        if distance <= 10:  # Distance check
            main_buildings[main_building_key]['sub_buildings'].append({'contour': building_contour})
    time.sleep(1)  # Add a pause between requests to avoid overloading the API

# Create a map
m = folium.Map(location=next(iter(main_buildings.values()))['coordinates'], zoom_start=18)

# Add buildings to the map
main_markers = []
sub_building_layers = {}
for index, (key, main_building) in enumerate(main_buildings.items()):
    # Add main building to the map
    main_popup = folium.Popup("Main Building", max_width=300)
    main_marker = folium.Marker(
        location=main_building['coordinates'],
        popup=main_popup,
        icon=folium.Icon(color='red'),
        tooltip="Click to highlight sub-buildings"
    )
    main_marker.add_to(m)
    main_markers.append(main_marker)

    # Create a layer for sub-buildings
    sub_building_layer = folium.FeatureGroup(name=f'Sub Buildings {index}')
    for sub_building in main_building['sub_buildings']:
        if sub_building['contour']:  # Check for contour presence
            folium.Polygon(
                locations=sub_building['contour'],
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0
            ).add_to(sub_building_layer)
    sub_building_layer.add_to(m)
    sub_building_layers[index] = sub_building_layer

# Save the map to an HTML file
m.save('buildings_map.html')

print(f"main_buildings {main_buildings}")
print(f"sub_building_layers {sub_building_layers}")
print(f"main_markers {main_markers}")

# Define JavaScript code for Leaflet
script = f"""
<script>
document.addEventListener('DOMContentLoaded', function() {{
    var map = L.map('map').setView({next(iter(main_buildings.values()))['coordinates']}, 18);

    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }}).addTo(map);

    var mainMarkers = [];
    var subBuildingLayers = {{}};

    {''.join([f'''
    var mainMarker{index} = L.marker({main_building['coordinates']}).addTo(map)
        .bindPopup('Main Building')
        .on('click', function() {{
            var layer = subBuildingLayers[{index}];
            if (map.hasLayer(layer)) {{
                map.removeLayer(layer);
            }} else {{
                map.addLayer(layer);
            }}
        }});
    mainMarkers.push(mainMarker{index});

    var subLayer{index} = L.layerGroup();
    {''.join([f"L.polygon({sub_building['contour']}, {{color: 'blue', fillColor: 'blue', fillOpacity: 0}}).addTo(subLayer{index});" for sub_building in main_building['sub_buildings'] if sub_building['contour']])}
    subBuildingLayers[{index}] = subLayer{index};
    
    var TPLayer = L.layerGroup();
    {''.join([f"L.polygon({main_building['main_building_border']}, {{color: 'blue', fillColor: 'blue', fillOpacity: 0.5, opacity: 1}}).addTo(TPLayer);"])}
    map.addLayer(TPLayer)

    ''' for index, main_building in enumerate(main_buildings.values())])}

    map.on('layeradd', function(e) {{
        if (e.layer instanceof L.Polygon) {{
            e.layer.setStyle({{
                fillOpacity: 0.5,
                fillColor: '#ffd800',
                opacity: 1,
                color: '#ffe866'
            }});
        }}
    }});

    map.on('layerremove', function(e) {{
        if (e.layer instanceof L.Polygon) {{
            e.layer.setStyle({{
                fillOpacity: 0,
                opacity: 0,
            }});
        }}
    }});
}});
</script>
"""

# Create the HTML file
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Map</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" crossorigin=""/>
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js" crossorigin=""></script>
    <style>
        #map {{
            width: 100%;
            height: 100vh;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    {script}
</body>
</html>
"""

with open('Карта.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Map successfully created and saved in buildings_map.html. Open this file in your web browser.")
