# Projekt

Der Code ist Teil meiner Bachelorarbeit. Die Hauptanalyse und das Ausführen der Anomalieerkennung werden in dem `analysis.ipynb` Notebook durchgeführt.

## Struktur

- `analysis_files\` enthält Abbildungen, die bereits in den Jupyter Notebooks erzeugt wurden.
- `Data\` enthält die Implementierung der Datenaufbereitung, des Autoencoders und der Anomalieerkennung, welche in dem `analysis.ipynb` Notebook aufgerufen werden. Weiterhin ist hier ein weiteres Notebook (`Data\document_log_content.ipynb`) gegeben, welches Analysen der ursprünglichen Daten vor der Datenaufbereitung enthält. In dem Ordner `Data\20130794\` ist weiterhin der verwendete Teil des Datensatzes (https://figshare.com/articles/dataset/Dataset_An_IoT-Enriched_Event_Log_for_Process_Mining_in_Smart_Factories/20130794) gegeben.
- `hyperparameters\` enthält die durch Optuna erstellten Hyperparameter, welche bei erneutem Training des Autoencoders wiederverwendet werden können.
- `loss_history\` enthält den Verlauf der Rekonstruktionsfehler während des Trainings (anhand des Validierungsdatensatzes).
- `model\` enthält die bereits trainierten Autoencoder-Modelle.
- `tables\` enthält Tabellen, die während der Analyse erzeugt wurden.
- `train_values\` enthält die zwischengespeicherten Ergebnisse der Datenvorverarbeitung.

## Ausführen des Codes

Alle benötigten Bibliotheken sind in der Datei `requirements.txt` gegeben. Das Umwandeln der ursprünglichen Daten in ein Parquet-Format erfolgt durch Ausführen von `Data\load_sensor_data.py`. Dies muss als Erstes ausgeführt werden, bevor die Anomalieerkennung durchgeführt werden kann. Die Datenvorverarbeitung, das Training des Autoencoders und die Anomalieerkennung werden durch das Ausführen von `analysis.ipynb` durchgeführt. Hier kann in dem zweiten Codeblock ausgewählt werden, ob die Datenvorverarbeitung, das Hyperparametertuning oder das Training erneut durchgeführt werden sollen, oder ob bereits erstellte Daten geladen werden sollen. Dabei ist zu beachten, dass das Hyperparametertuning sehr zeitaufwändig ist. Um die hohe Datenmenge effizient verarbeiten zu können (vor allem bei der Datenvorverarbeitung), wird außerdem vorgeschlagen, ausreichend Arbeitsspeicher zu verwenden (z. B. 32 GB).


