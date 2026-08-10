import csv
import utm
import os

input_file = os.path.expanduser('~/ros2_ws/src/auto_car_test_pkg/auto_car_test_pkg/waypoints/waypoints_20260304_160208.csv')
output_file = os.path.expanduser('~/ros2_ws/src/auto_car_test_pkg/auto_car_test_pkg/waypoints/waypoints_20260304_160208.kml')

utm_zone_number = 52
utm_zone_letter = 'N'

# KML 헤더
kml_header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>Waypoints</name>
    <Style id="yellowLineGreenPoly">
        <LineStyle>
            <color>7f00ffff</color>
            <width>4</width>
        </LineStyle>
        <PolyStyle>
            <color>7f00ff00</color>
        </PolyStyle>
    </Style>
    <Placemark>
        <name>Path</name>
        <styleUrl>#yellowLineGreenPoly</styleUrl>
        <LineString>
            <coordinates>
"""

kml_footer = """
            </coordinates>
        </LineString>
    </Placemark>
</Document>
</kml>
"""

with open(input_file, newline='') as fin, open(output_file, 'w') as fout:
    reader = csv.DictReader(fin)
    fout.write(kml_header)
    
    for row in reader:
        x = float(row['utm_x'])
        y = float(row['utm_y'])
        flag = row['flag']
        lat, lon = utm.to_latlon(x, y, utm_zone_number, utm_zone_letter)
        fout.write(f"{lon},{lat},0\n")  # KML: lon,lat,altitude
    
    fout.write(kml_footer)

print(f"KML 변환 완료! {output_file} 파일을 Google Earth에서 열 수 있습니다.")