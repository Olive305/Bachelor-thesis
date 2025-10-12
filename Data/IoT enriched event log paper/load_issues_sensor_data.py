import os
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pyarrow as pa
import glob
import pyarrow.parquet as pq

def parse_datastream_from_event_xml(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    ns = {
        "xes": "http://code.deckfour.org/xes",
        "stream": "https://cpee.org/datastream/datastream.xesext"
    }

    results = []

    # Iterate over all <event> elements
    for event_xml in root.findall(".//xes:event", ns):
        base = {}

        # Extract top-level attributes like concept:name, org:resource, SubProcessID
        for attr in event_xml.findall("xes:string", ns):
            key = attr.attrib.get("key")
            val = attr.attrib.get("value")
            if key in ("concept:name", "org:resource", "SubProcessID"):
                base[key] = val

        # Find datastream lists
        datastreams = event_xml.findall("xes:list[@key='stream:datastream']", ns)
        if not datastreams:
            continue

        for datastream in datastreams:
            for idx, point in enumerate(datastream.findall("xes:list", ns)):
                row = base.copy()

                # Extract stream attributes (namespaced)
                row["stream:system"] = point.attrib.get(f"{{{ns['stream']}}}system")
                row["stream:system_type"] = point.attrib.get(f"{{{ns['stream']}}}system_type")
                row["stream:observation"] = point.attrib.get(f"{{{ns['stream']}}}observation")
                row["stream:procedure_type"] = point.attrib.get(f"{{{ns['stream']}}}procedure_type")
                row["stream:interaction_type"] = point.attrib.get(f"{{{ns['stream']}}}interaction_type")

                # Extract timestamp and value from children
                timestamp = None
                value = None
                for child in point:
                    if child.tag == f"{{{ns['xes']}}}date":
                        timestamp = child.attrib.get(f"{{{ns['stream']}}}timestamp")
                    elif child.tag == f"{{{ns['xes']}}}string":
                        value = child.attrib.get(f"{{{ns['stream']}}}value")

                row["stream:timestamp"] = timestamp
                row["stream:value"] = value

                # Add a sensor key if needed
                row["sensor_key"] = point.attrib.get("key", f"stream:point_{idx}")

                results.append(row)

    return results

def process_file(filename, parquet_dir):
    try:
        parquet_file = os.path.join(parquet_dir, Path(filename).stem + ".parquet")
        sensor_info = parse_datastream_from_event_xml(filename)
        table = pa.Table.from_pylist(sensor_info)
        pq.write_table(table, parquet_file, compression="snappy")
        return f"Processed {filename}"
    except Exception as e:
        return f"Error processing {filename}: {e}"

if __name__ == "__main__":
    
    xes_directory = Path.cwd() / "Data" / "IoT enriched event log paper" / "20130794" / "Data Quality Issues Event Log"
    xes_files = [f for f in xes_directory.glob("*.xes") if f.name != "MainProcess.xes"]
    print(f"Found {len(xes_files)} .xes files.")

    parquet_dir = xes_directory / "parquet"
    parquet_dir.mkdir(exist_ok=True)
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, str(f), str(parquet_dir)): f for f in xes_files}
        for i, future in enumerate(as_completed(futures), 1):
            print(f"Progress: {i}/{len(xes_files)} — {future.result()}")

    print("All files processed.")
    
    # Combine all parquet files into one

    parquet_files = sorted(glob.glob(str(parquet_dir / "*.parquet")))
    tables = []
    for f in parquet_files:
        try:
            table = pq.read_table(f)
            tables.append(table)
        except Exception as e:
            print(f"Skipping {f} due to schema error: {e}")
    if tables:
        try:
            combined_table = pa.concat_tables(tables, promote=True)
            combined_file = parquet_dir / "all_combined.parquet"
            pq.write_table(combined_table, combined_file, compression="snappy")
            print(f"Combined {len(tables)} files into {combined_file}")
        except Exception as e:
            print(f"Error combining tables: {e}")
    else:
        print("No valid parquet files found to combine.")
    