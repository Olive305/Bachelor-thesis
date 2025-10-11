import os
import pandas as pd
from lxml import etree
import xml.etree.ElementTree as ET
import duckdb
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
from typing import Optional

# Helper: parse datastream list element into a dict structure
def parse_datastream_from_event_xml(filename):
    
    tree = ET.parse(filename)
    root = tree.getroot()
    # Namespace for XES
    ns = {"xes": "http://code.deckfour.org/xes"}
    # Find all event elements
    event_xml_list = root.findall(".//xes:event", ns)
    results = []
    for event_xml in event_xml_list:
        # The rest of your code (from # Namespace for XES onwards) should be indented and placed here,
        # replacing 'event_xml' with the current event_xml in the loop.
        datastreams = event_xml.findall("xes:list[@key='stream:datastream']", ns)
        if not datastreams:
            continue

        merged = {"children": {}}
        for datastream in datastreams:
            for idx, point in enumerate(datastream.findall("xes:list", ns)):
                stream_ns = "https://cpee.org/datastream/datastream.xesext"
                sensor_point = {
                    "stream:system": point.attrib.get(f"{{{stream_ns}}}system"),
                    "stream:system_type": point.attrib.get(f"{{{stream_ns}}}system_type"),
                    "stream:observation": point.attrib.get(f"{{{stream_ns}}}observation"),
                    "stream:procedure_type": point.attrib.get(f"{{{stream_ns}}}procedure_type"),
                    "stream:interaction_type": point.attrib.get(f"{{{stream_ns}}}interaction_type"),
                    "children": {}
                }
                for child in point:
                    key = child.attrib.get("key")
                    val = child.attrib.get("value")
                    if key:
                        sensor_point["children"][key] = val
                sensor_key = point.attrib.get("key", f"stream:point_{idx}")
                if sensor_key in merged["children"]:
                    i = 1
                    new_key = f"{sensor_key}_{i}"
                    while new_key in merged["children"]:
                        i += 1
                        new_key = f"{sensor_key}_{i}"
                    sensor_key = new_key
                merged["children"][sensor_key] = sensor_point
        results.append(merged)
    return results
    

def process_xes_to_parquet(file):
    df = parse_datastream_from_event_xml(file)
    # Save each file's result as a parquet file (or handle as needed)
    out_file = os.path.splitext(file)[0] + ".parquet"
    pd.DataFrame(df).to_parquet(out_file)

if __name__ == "__main__":
    # Directory with your .xes files
    xes_directory = os.path.join(os.getcwd(), "Data", "IoT enriched event log paper", "20130794", "Cleaned Event Log")
    xes_files = [
        os.path.join(xes_directory, f)
        for f in os.listdir(xes_directory)
        if f.endswith('.xes') and f != "MainProcess.xes"
    ]
    print(f"Found {len(xes_files)} .xes files to process.")

    with ProcessPoolExecutor() as executor:
        executor.map(process_xes_to_parquet, xes_files)