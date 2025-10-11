import os
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

def parse_datastream_from_event_xml(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    ns = {"xes": "http://code.deckfour.org/xes"}
    event_xml_list = root.findall(".//xes:event", ns)
    results = []
    for event_xml in event_xml_list:
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
                merged["children"][sensor_key] = sensor_point
        results.append(merged)
    return results

def process_file(filename, parquet_dir):
    try:
        parquet_file = os.path.join(parquet_dir, Path(filename).stem + ".parquet")
        if os.path.exists(parquet_file):
            return f"Skipping {filename}, Parquet already exists."
        sensor_info = parse_datastream_from_event_xml(filename)
        table = pa.Table.from_pylist(sensor_info)
        pq.write_table(table, parquet_file, compression="snappy")
        return f"Processed {filename}"
    except Exception as e:
        return f"Error processing {filename}: {e}"

if __name__ == "__main__":
    xes_directory = Path.cwd() / "Data" / "IoT enriched event log paper" / "20130794" / "Cleaned Event Log"
    xes_files = [f for f in xes_directory.glob("*.xes") if f.name != "MainProcess.xes"]
    print(f"Found {len(xes_files)} .xes files.")

    parquet_dir = xes_directory / "parquet"
    parquet_dir.mkdir(exist_ok=True)

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, str(f), str(parquet_dir)): f for f in xes_files}
        for i, future in enumerate(as_completed(futures), 1):
            print(f"Progress: {i}/{len(xes_files)} — {future.result()}")

    print("All files processed.")
