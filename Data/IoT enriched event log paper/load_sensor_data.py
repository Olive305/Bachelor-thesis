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
    ns = {"xes": "http://code.deckfour.org/xes"}
    event_xml_list = root.findall(".//xes:event", ns)
    results = []

    # Pre-extract all traces and their attributes
    trace_attrs_map = {}
    for trace in root.findall(".//xes:trace", ns):
        trace_attrs = {}
        for attr in trace.findall("*", ns):
            key = attr.attrib.get("key")
            val = attr.attrib.get("value")
            if key:
                trace_attrs[key] = val
        trace_id = id(trace)
        trace_attrs_map[trace_id] = trace_attrs

        # Mark all events in this trace with a reference to their trace attributes
        for event in trace.findall("xes:event", ns):
            event.attrib["_trace_id"] = str(trace_id)

    for event_xml in event_xml_list:
        # Extract all event attributes (include all types)
        event_attrs = {}
        for attr in event_xml.findall("*", ns):
            key = attr.attrib.get("key")
            val = attr.attrib.get("value")
            if key:
                event_attrs[key] = val

        # Get trace attributes for this event
        trace_id = int(event_xml.attrib.get("_trace_id", "0"))
        trace_attrs = trace_attrs_map.get(trace_id, {})

        # Merge trace and event attributes (trace attributes prefixed)
        base = {f"trace:{k}": v for k, v in trace_attrs.items()}
        base.update(event_attrs)

        datastreams = event_xml.findall("xes:list[@key='stream:datastream']", ns)
        if not datastreams:
            continue
        for datastream in datastreams:
            for idx, point in enumerate(datastream.findall("xes:list", ns)):
                stream_ns = "https://cpee.org/datastream/datastream.xesext"
                row = base.copy()
                # Flatten top-level attributes
                row["stream:system"] = point.attrib.get(f"{{{stream_ns}}}system")
                row["stream:system_type"] = point.attrib.get(f"{{{stream_ns}}}system_type")
                row["stream:observation"] = point.attrib.get(f"{{{stream_ns}}}observation")
                row["stream:procedure_type"] = point.attrib.get(f"{{{stream_ns}}}procedure_type")
                row["stream:interaction_type"] = point.attrib.get(f"{{{stream_ns}}}interaction_type")
                # Flatten children
                for child in point:
                    key = child.attrib.get("key")
                    val = child.attrib.get("value")
                    if key:
                        row[key] = val
                # Optionally, add the sensor key if needed
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
    xes_directory = Path.cwd() / "Data" / "IoT enriched event log paper" / "20130794" / "Cleaned Event Log"
    xes_files = [f for f in xes_directory.glob("*.xes") if f.name != "MainProcess.xes"]
    print(f"Found {len(xes_files)} .xes files.")

    parquet_dir = xes_directory / "parquet"
    parquet_dir.mkdir(exist_ok=True)
    
    if True:
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_file, str(f), str(parquet_dir)): f for f in xes_files}
            for i, future in enumerate(as_completed(futures), 1):
                print(f"Progress: {i}/{len(xes_files)} — {future.result()}")

        print("All files processed.")
    
    # Combine all parquet files into one

    parquet_files = sorted(glob.glob(str(parquet_dir / "*.parquet")))
    combined_file = parquet_dir / "all_combined_new.parquet"
    writer = None
    num_tables = 0
    print(f"Combining {len(parquet_files)} parquet files into {combined_file}...")
    for i, f in enumerate(parquet_files, 1):
        try:
            print(f"[{i}/{len(parquet_files)}] Reading {f}...")
            table = pq.read_table(f)
            if writer is None:
                writer = pq.ParquetWriter(combined_file, table.schema, compression="snappy")
                print(f"Initialized ParquetWriter with schema from {f}")
            writer.write_table(table)
            num_tables += 1
            print(f"Written {f} to combined file.")
        except Exception as e:
            print(f"Skipping {f} due to schema error: {e}")
    if writer:
        writer.close()
        print(f"Combined {num_tables} files into {combined_file}")
    else:
        print("No valid parquet files found to combine.")
    