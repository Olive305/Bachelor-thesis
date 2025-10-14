## Project Structure

This repository contains code and part of the data for my bachelor thesis on Data/IoT-enriched event logs.

Event log files (`.xes`) are not included because of data size.

### Data

- **Event logs**  
    Event logs are not uploaded to GitHub, but are in the `Data\IoT enriched event log paper\20130794\Cleaned Event Log` folder

- **Analysis code**  
    Given in a Jupyter Notebook in `Data\IoT enriched event log paper\document_log_content.ipynb`

- **Sensor data extraction**  
    Given in the python file `Data\IoT enriched event log paper\load_sensor_data.py`. The `.parquet` files are stored in the `Data\IoT enriched event log paper\20130794\Cleaned Event Log\parquet` folder and can be queried using SQL commands in DuckDB (Not yet uploaded because of data size issues)
    `Data\IoT enriched event log paper\load_issues_sensor_data.py` extracts the data with the quality issues

### Run the code

A `requirements.txt` file has not yet been created, but will be added shortly.

Python 3 is used.
