import duckdb
import pandas as pd
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score

def prepare_ov_1_data(file_location):
    # Load your data from the specified file location
    # DuckDB connection
    con = duckdb.connect()

    # Create a df with all the values of temperature sensors sorted by their timestamp
    #! Fix column names in query
    df = con.execute(f"""
        SELECT 
        "stream:timestamp",
        "concept:name",
        "stream:system" || '_' || "stream:observation" AS "sensor:id",
        "stream:value"
        FROM read_parquet('{file_location}')
        WHERE "org:resource" = 'ov_1'
        AND "stream:observation" NOT LIKE '%NFC%'
        AND "stream:observation" NOT LIKE '%Current_State%'
        GROUP BY "stream:observation", "stream:system", "stream:value", "stream:timestamp", "concept:name"
        ORDER BY "stream:timestamp" ASC
    """).df()
    # ensure timestamp column is actual datetime objects for safe comparison with datetime.datetime
    df['stream:timestamp'] = pd.to_datetime(df['stream:timestamp'])
    
    print(f"    Loaded {len(df)} rows from {file_location}\n")
    
    print(" Data loaded. Reshaping...")

    # Change the data to include same timestamps in a single row
    # Create a new dataframe

    """
    Now we assign sensor values to timestamps

    How:
    Since a large portion of the data is already synchronized, we can use the time intervals given by 
    these synchronized values and from other sensors we just add the closest value
    """

    # One of the synchronized sensors, which will be used to get the timestamps
    reference_sensor_id = "http://iot.uni-trier.de/FTOnto#OV_1_WT_1_Compressor_8_http://iot.uni-trier.de/FTOnto#CompressorPowerLevel"

    #? Is the ref dataframe directly accessible as id? So can we just copy values to the other df like here?
    # Split reference and others
    ref = df[df["sensor:id"] == reference_sensor_id].sort_values("stream:timestamp")
    others = df[df["sensor:id"] != reference_sensor_id].sort_values("stream:timestamp")

    # Reset index for safe alignment and display the reference dataframe
    ref = ref.reset_index(drop=True)
    print("Reference dataframe preview (first 10 rows):")
    print(ref.head(10))
    print(f"Reference dataframe shape: {ref.shape}\n")

    # Initialize 'wide' with the reference timestamps and the reference sensor values
    wide = pd.DataFrame({
        "stream:timestamp": ref["stream:timestamp"].values,
        reference_sensor_id: ref["stream:value"].values
    })

    # Show the initialized wide head so you can confirm alignment before merging others
    print("Initialized 'wide' preview:")
    print(wide.head(10))

    events = df["concept:name"].unique().tolist() # Binary value per event to indicate which event is happening

    # Create binary columns for each event
    for event in events:
        wide[f"event:{event}"] = (ref["concept:name"] == event).astype(int).values

    print(" Initial sensor added with timestamps and event data. Merging other sensors...")
    
    for sensor_id in others["sensor:id"].unique():
        sensor_data = others[others["sensor:id"] == sensor_id][["stream:timestamp", "stream:value"]].reset_index(drop=True)
        
        print(f"  Merging sensor: {sensor_id} with {len(sensor_data)} entries")
        
        # For each timestamp in wide, find the closest timestamp in sensor_data
        sensor_values = []
        sensor_timestamps = sensor_data["stream:timestamp"].values
        sensor_values_list = sensor_data["stream:value"].values
        
        print(f"\n   Searching closest timestamps for {len(wide)} entries...")
        
        for ts in wide["stream:timestamp"]:
            # Find index of closest timestamp
            pos = sensor_data["stream:timestamp"].searchsorted(ts)

            if pos == 0:
                sensor_values.append(sensor_values_list[0])
            elif pos >= len(sensor_timestamps):
                sensor_values.append(sensor_values_list[-1])
            else:
                left_ts = pd.Timestamp(sensor_timestamps[pos - 1])
                right_ts = pd.Timestamp(sensor_timestamps[pos])
                # choose the closer timestamp
                if abs(ts - left_ts) <= abs(right_ts - ts):
                    sensor_values.append(sensor_values_list[pos - 1])
                else:
                    sensor_values.append(sensor_values_list[pos])
                    
        print(f"   Merged {len(sensor_values)} values for sensor {sensor_id}")
        
        wide[sensor_id] = sensor_values

    return wide

def detect_column_types(df):
    cont_idx = []
    bin_idx = []
    evt_idx = []

    for i, col in enumerate(df.columns):
        
        # Timestamps are NOT model features → ignore as early as possible
        if "timestamp" in col:
            continue

        # Event columns start with 'event:'
        if col.startswith("event:"):
            evt_idx.append(i)
            continue

        series = df[col]
        print(f" Sample values: {series.unique()[:5]}")
        unique_vals = series.dropna().unique()

        # Robust binary detection:
        #  - Try to coerce unique values to numeric and check subset of {0.0,1.0}
        #  - Treat boolean dtype as binary
        num_vals = pd.to_numeric(pd.Series(unique_vals), errors="coerce").dropna().unique()
        print(f" Unique values: {unique_vals}, Numeric values: {num_vals}")
        if len(num_vals) > 0 and set(num_vals).issubset({0.0, 1.0}):
            bin_idx.append(i)
        elif pd.api.types.is_bool_dtype(series):
            bin_idx.append(i)
        else:
            # Everything else = continuous
            cont_idx.append(i)

    return cont_idx, bin_idx, evt_idx

def prepare_data_for_bayesian_network(df):
    print(" Detecting column types...")
    cont_idx, bin_idx, evt_idx = detect_column_types(df)
    print(f" Detected {len(cont_idx)} continuous, {len(bin_idx)} binary, and {len(evt_idx)} event columns.")

    # Prepare data subsets
    cont_data = df.iloc[:, cont_idx].reset_index(drop=True)
    bin_data = df.iloc[:, bin_idx].reset_index(drop=True)
    evt_data = df.iloc[:, evt_idx].reset_index(drop=True)

    return cont_data, bin_data, evt_data

def discretize_continuous(cont_data, n_bins=5, strategy="quantile"):
    if cont_data is None or cont_data.empty:
        return pd.DataFrame(index=[])
    kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    disc = kb.fit_transform(cont_data.fillna(method="ffill").fillna(method="bfill"))
    disc_df = pd.DataFrame(disc, columns=cont_data.columns, index=cont_data.index).astype(int)
    return disc_df

def _to_int_df_safe(df):
    """
    Safely coerce a DataFrame to integer values:
    - Coerce each column to numeric (errors -> NaN)
    - Fill NaNs with 0 (safe default for binary/event)
    - Round to nearest integer then astype(int)
    """
    if df is None or df.empty:
        return pd.DataFrame(index=df.index if df is not None else [])
    df_numeric = df.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    df_filled = df_numeric.fillna(0)
    df_rounded = df_filled.round().astype(int)
    df_rounded.columns = df.columns
    df_rounded.index = df.index
    return df_rounded

def combine_discrete(bin_data, evt_data, disc_cont):
    parts = []
    if not (bin_data is None or bin_data.empty):
        parts.append(_to_int_df_safe(bin_data))
    if not (evt_data is None or evt_data.empty):
        parts.append(_to_int_df_safe(evt_data))
    if not (disc_cont is None or disc_cont.empty):
        parts.append(_to_int_df_safe(disc_cont))
    if not parts:
        return pd.DataFrame(index=[])
    return pd.concat(parts, axis=1)

def build_chow_liu_tree(discrete_df):
    cols = list(discrete_df.columns)
    G = nx.Graph()
    G.add_nodes_from(cols)
    # Compute pairwise mutual information
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            x = discrete_df[cols[i]].values
            y = discrete_df[cols[j]].values
            w = mutual_info_score(x, y)
            if np.isfinite(w) and w > 0:
                G.add_edge(cols[i], cols[j], weight=w)
    # Maximum Spanning Tree
    if G.number_of_edges() == 0:
        return G
    T = nx.maximum_spanning_tree(G, weight="weight")
    return T

def estimate_univariate_probs(discrete_df, alpha=1.0):
    probs = {}
    for col in discrete_df.columns:
        counts = discrete_df[col].value_counts().sort_index()
        support = np.arange(discrete_df[col].min(), discrete_df[col].max() + 1)
        counts = counts.reindex(support, fill_value=0)
        total = counts.sum()
        p = (counts + alpha) / (total + alpha * len(support))
        probs[(col,)] = p.to_dict()
    return probs

def estimate_pairwise_probs(discrete_df, tree_graph, alpha=1.0):
    cond_probs = {}
    for u, v in tree_graph.edges():
        xu = discrete_df[u]
        xv = discrete_df[v]
        # Build contingency table over observed support
        u_vals = np.arange(xu.min(), xu.max() + 1)
        v_vals = np.arange(xv.min(), xv.max() + 1)
        table = pd.crosstab(xu, xv).reindex(index=u_vals, columns=v_vals, fill_value=0)
        # P(v|u)
        row_sums = table.sum(axis=1)
        p_v_given_u = ((table + alpha).div(row_sums + alpha * len(v_vals), axis=0)).to_dict()
        cond_probs[(u, v)] = p_v_given_u
        # Also store P(u|v) for symmetric scoring
        col_sums = table.sum(axis=0)
        p_u_given_v = ((table + alpha).div(col_sums + alpha * len(u_vals), axis=1)).T.to_dict()
        cond_probs[(v, u)] = p_u_given_v
    return cond_probs

def pick_root(tree_graph):
    if tree_graph.number_of_nodes() == 0:
        return None
    # Root = node with highest degree
    return max(tree_graph.degree, key=lambda x: x[1])[0]

def score_samples_chow_liu(discrete_df, tree_graph, uni_probs, cond_probs):
    root = pick_root(tree_graph)
    log_scores = []
    for _, row in discrete_df.iterrows():
        logp = 0.0
        # Root probability
        if root is not None:
            rv = int(row[root])
            p_root_dict = uni_probs.get((root,), {})
            p_root = p_root_dict.get(rv, min(p_root_dict.values()) if p_root_dict else 1e-12)
            logp += np.log(p_root + 1e-12)
        # Edge conditionals
        for u, v in tree_graph.edges():
            # Score v|u and u|v to be robust (sum logs)
            u_val = int(row[u])
            v_val = int(row[v])
            pv_given_u = cond_probs.get((u, v), {}).get(u_val, {})
            pu_given_v = cond_probs.get((v, u), {}).get(v_val, {})
            logp += np.log(pv_given_u.get(v_val, 1e-12) + 1e-12)
            logp += np.log(pu_given_v.get(u_val, 1e-12) + 1e-12)
        log_scores.append(logp)
    return np.array(log_scores)

def detect_anomalies(log_scores, threshold_quantile=0.05):
    thr = np.quantile(log_scores, threshold_quantile)
    anomalies = log_scores <= thr
    return anomalies, thr

def _format_node_label(label: str, wrap_width: int = 18) -> str:
    """
    Make long labels readable by:
    - Removing long common prefixes (e.g., ontology base URLs)
    - Keeping only the tail after the last '#', '/' or '_http://...#'
    - Wrapping to multiple lines with the given width
    """
    base_prefixes = [
        "http://iot.uni-trier.de/FTOnto#",
        "https://iot.uni-trier.de/FTOnto#",
    ]
    s = label
    for p in base_prefixes:
        if s.startswith(p):
            s = s[len(p):]
            break

    # If label contains another embedded URI, keep the tail after the last '#'
    if "#" in s:
        s = s.split("#")[-1]
    # Collapse long path-like labels to last segment
    if "/" in s:
        s = s.split("/")[-1]

    # For very long single tokens, insert soft wraps every wrap_width chars
    import textwrap
    s = "\n".join(textwrap.wrap(s, width=wrap_width)) if len(s) > wrap_width else s
    return s

def visualize_sensor_relations(tree_graph, save_path=None, wrap_width: int = 18):
    # Dynamically scale figure size based on node count to reduce crowding
    n = max(tree_graph.number_of_nodes(), 1)
    width = min(max(14, int(n * 0.9)), 48)
    height = min(max(9, int(n * 0.6)), 36)
    plt.figure(figsize=(width, height))

    # Increase optimal distance between nodes to space them out more
    # Use a larger k and a bigger scale so nodes are placed farther apart
    k = 20.0 / np.sqrt(n)  # larger than before to push nodes apart
    pos = nx.spring_layout(tree_graph, k=k, iterations=800, seed=42, scale=3.0)

    # Optionally add slight jitter to avoid exact overlaps for very similar positions
    jitter = 0.02
    for node in pos:
        pos[node] = (pos[node][0] + np.random.uniform(-jitter, jitter),
                     pos[node][1] + np.random.uniform(-jitter, jitter))

    labels = {node: _format_node_label(node, wrap_width) for node in tree_graph.nodes()}
    
    # Extract edge weights (mutual information scores) and normalize for visualization
    edge_weights = nx.get_edge_attributes(tree_graph, 'weight')
    if edge_weights:
        max_weight = max(edge_weights.values())
        min_weight = min(edge_weights.values())
        # Normalize edge widths: thicker = stronger relationship
        edge_widths = [1 + 4 * (edge_weights.get((u, v), 0) - min_weight) / (max_weight - min_weight + 1e-10)
                       for u, v in tree_graph.edges()]
    else:
        edge_widths = [1.0] * tree_graph.number_of_edges()
    
    # Draw nodes and edges
    nx.draw(
        tree_graph,
        pos,
        with_labels=True,
        labels=labels,
        node_size=1800,
        node_color="lightblue",
        font_size=9,
        font_weight="bold",
        edge_color="gray",
        width=edge_widths
    )
    
    # Draw edge labels showing mutual information values
    if edge_weights:
        edge_labels = {(u, v): f"{w:.3f}" for (u, v), w in edge_weights.items()}
        nx.draw_networkx_edge_labels(
            tree_graph,
            pos,
            edge_labels=edge_labels,
            font_size=7,
            font_color="darkred",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.7)
        )

    # Add margins so labels have breathing room
    plt.margins(0.15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

def extract_correlation_rules(cont_data, threshold=0.5):
    """
    Extract correlation rules from continuous data.
    Returns rules like: "If sensor A increases, sensor B also increases"
    """
    if cont_data is None or cont_data.empty or len(cont_data.columns) < 2:
        return []
    
    corr_matrix = cont_data.corr(method='spearman')
    rules = []
    
    for i, col_a in enumerate(corr_matrix.columns):
        for j, col_b in enumerate(corr_matrix.columns):
            if i < j:  # Only upper triangle
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    direction = "increases" if corr > 0 else "decreases"
                    rule = {
                        "sensor_a": col_a,
                        "sensor_b": col_b,
                        "correlation": corr,
                        "rule": f"When {_format_node_label(col_a, 30)} increases, {_format_node_label(col_b, 30)} {direction}",
                        "strength": abs(corr)
                    }
                    rules.append(rule)
    
    # Sort by strength
    rules.sort(key=lambda x: x["strength"], reverse=True)
    return rules

def extract_conditional_rules(discrete_df, tree_graph, cond_probs, top_k=20):
    """
    Extract conditional probability rules from the Bayesian Network.
    Returns rules like: "If sensor A = 2, then sensor B = 3 (prob: 0.85)"
    """
    rules = []
    
    for (parent, child), prob_dict in cond_probs.items():
        # Only process parent->child direction (skip reverse)
        if (parent, child) not in tree_graph.edges():
            continue
            
        # For each parent value, find the most likely child value
        for parent_val, child_dist in prob_dict.items():
            if not child_dist:
                continue
            # Find most probable child value given parent value
            most_likely_child = max(child_dist.items(), key=lambda x: x[1])
            child_val, prob = most_likely_child
            
            # Only include strong rules (high probability)
            if prob > 0.6:
                rule = {
                    "parent": parent,
                    "parent_value": parent_val,
                    "child": child,
                    "child_value": child_val,
                    "probability": prob,
                    "rule": f"If {_format_node_label(parent, 30)} = {parent_val}, then {_format_node_label(child, 30)} = {child_val} (prob: {prob:.2f})",
                }
                rules.append(rule)
    
    # Sort by probability
    rules.sort(key=lambda x: x["probability"], reverse=True)
    return rules[:top_k]

def extract_multivariate_rules(discrete_df, tree_graph, cond_probs, top_k=10):
    """
    Extract complex rules involving multiple sensors.
    """
    rules = []
    
    # Find nodes with degree > 1 (hub nodes)
    hubs = [node for node, degree in tree_graph.degree() if degree > 1]
    
    for hub in hubs:
        neighbors = list(tree_graph.neighbors(hub))
        if len(neighbors) >= 2:
            # Create a rule: if hub has certain value, what are likely neighbor values
            hub_vals = discrete_df[hub].unique()
            
            for hub_val in hub_vals:
                mask = discrete_df[hub] == hub_val
                if mask.sum() < 10:  # Skip rare cases
                    continue
                
                # Find most common neighbor values when hub = hub_val
                neighbor_modes = {}
                for neighbor in neighbors[:3]:  # Limit to 3 neighbors
                    mode_val = discrete_df.loc[mask, neighbor].mode()
                    if len(mode_val) > 0:
                        neighbor_modes[neighbor] = mode_val.iloc[0]
                
                if len(neighbor_modes) >= 2:
                    freq = mask.sum() / len(discrete_df)
                    rule_parts = [f"{_format_node_label(n, 25)} = {v}" for n, v in neighbor_modes.items()]
                    rule = {
                        "hub": hub,
                        "hub_value": hub_val,
                        "neighbors": neighbor_modes,
                        "frequency": freq,
                        "rule": f"When {_format_node_label(hub, 30)} = {hub_val}, typically: {', '.join(rule_parts)} (freq: {freq:.2%})"
                    }
                    rules.append(rule)
    
    rules.sort(key=lambda x: x["frequency"], reverse=True)
    return rules[:top_k]

def save_rules_to_file(rules_dict, output_path):
    """Save extracted rules to a text file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EXTRACTED SENSOR RELATIONSHIP RULES\n")
        f.write("=" * 80 + "\n\n")
        
        if "correlation_rules" in rules_dict and rules_dict["correlation_rules"]:
            f.write("\n" + "=" * 80 + "\n")
            f.write("1. CORRELATION RULES (Continuous Relationships)\n")
            f.write("=" * 80 + "\n")
            f.write("These rules show how sensors move together in their continuous values.\n\n")
            for i, rule in enumerate(rules_dict["correlation_rules"], 1):
                f.write(f"{i}. {rule['rule']}\n")
                f.write(f"   Correlation coefficient: {rule['correlation']:.3f}\n\n")
        
        if "conditional_rules" in rules_dict and rules_dict["conditional_rules"]:
            f.write("\n" + "=" * 80 + "\n")
            f.write("2. CONDITIONAL PROBABILITY RULES (If-Then Patterns)\n")
            f.write("=" * 80 + "\n")
            f.write("These rules show likely sensor states given another sensor's state.\n\n")
            for i, rule in enumerate(rules_dict["conditional_rules"], 1):
                f.write(f"{i}. {rule['rule']}\n\n")
        
        if "multivariate_rules" in rules_dict and rules_dict["multivariate_rules"]:
            f.write("\n" + "=" * 80 + "\n")
            f.write("3. MULTIVARIATE RULES (Complex Patterns)\n")
            f.write("=" * 80 + "\n")
            f.write("These rules show common patterns across multiple sensors.\n\n")
            for i, rule in enumerate(rules_dict["multivariate_rules"], 1):
                f.write(f"{i}. {rule['rule']}\n\n")

def build_bn_and_detect_anomalies(df, n_bins=5, quantile=0.05, save_graph_to=None, extract_rules=True, rules_output_path=None):
    cont_data, bin_data, evt_data = prepare_data_for_bayesian_network(df)
    disc_cont = discretize_continuous(cont_data, n_bins=n_bins)
    discrete_df = combine_discrete(bin_data, evt_data, disc_cont)
    tree = build_chow_liu_tree(discrete_df)
    uni_probs = estimate_univariate_probs(discrete_df)
    cond_probs = estimate_pairwise_probs(discrete_df, tree)
    log_scores = score_samples_chow_liu(discrete_df, tree, uni_probs, cond_probs)
    anomalies, thr = detect_anomalies(log_scores, threshold_quantile=quantile)
    
    if save_graph_to:
        visualize_sensor_relations(tree, save_path=save_graph_to)
    
    # Extract rules if requested
    rules_dict = {}
    if extract_rules:
        print("\nExtracting relationship rules...")
        rules_dict["correlation_rules"] = extract_correlation_rules(cont_data, threshold=0.5)
        rules_dict["conditional_rules"] = extract_conditional_rules(discrete_df, tree, cond_probs, top_k=20)
        rules_dict["multivariate_rules"] = extract_multivariate_rules(discrete_df, tree, cond_probs, top_k=10)
        
        if rules_output_path:
            save_rules_to_file(rules_dict, rules_output_path)
            print(f"Rules saved to: {rules_output_path}")
    
    result = pd.DataFrame({
        "log_score": log_scores,
        "is_anomaly": anomalies
    }, index=df.index)
    return {
        "tree": tree,
        "scores": result,
        "threshold": thr,
        "discrete_df": discrete_df,
        "rules": rules_dict
    }

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run Bayesian Network anomaly detection on IoT event log data")
    parser.add_argument("--bins", type=int, default=5, help="Number of bins for discretizing continuous variables")
    parser.add_argument("--quantile", type=float, default=0.05, help="Quantile threshold for anomalies (e.g., 0.05)")
    parser.add_argument("--save-graph", type=str, default=None, help="Optional path to save the relations graph PNG")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to write outputs (scores CSV, graph)")
    parser.add_argument("--extract-rules", action="store_true", help="Extract and save interpretable rules")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_path = args.save_graph if args.save_graph else str(out_dir / "sensor_relations.png")
    rules_path = str(out_dir / "sensor_rules.txt") if args.extract_rules else None

    print("Loading and preparing data...")
    # Get file location
    parquet_directory = os.path.join(os.getcwd(), "Data", "IoT enriched event log paper", "20130794", "Cleaned Event Log", "parquet")
    file_location = os.path.join(parquet_directory, "all_combined_with_synthetic.parquet")
    df_wide = prepare_ov_1_data(file_location)  # removed unused self

    print("Building Bayesian Network and detecting anomalies...")
    result = build_bn_and_detect_anomalies(
        df_wide,
        n_bins=args.bins,
        quantile=args.quantile,
        save_graph_to=graph_path,
        extract_rules=args.extract_rules,
        rules_output_path=rules_path
    )

    # Save scores
    scores_csv = out_dir / "bn_anomaly_scores.csv"
    result["scores"].to_csv(scores_csv, index=False)

    # Basic outputs
    print("\n=== Bayesian Network Results ===")
    print(f"Variables: {len(result['discrete_df'].columns)}")
    print(f"Edges in relations tree: {result['tree'].number_of_edges()}")
    print(f"Log-likelihood threshold (quantile {args.quantile}): {result['threshold']:.6f}")
    print(f"Anomalies flagged: {int(result['scores']['is_anomaly'].sum())} / {len(result['scores'])}")
    print(f"Scores saved to: {scores_csv}")
    print(f"Relations graph saved to: {graph_path}")
    
    # Print sample rules if extracted
    if args.extract_rules and "rules" in result:
        print("\n=== Sample Extracted Rules ===")
        if result["rules"].get("correlation_rules"):
            print("\nTop 3 Correlation Rules:")
            for i, rule in enumerate(result["rules"]["correlation_rules"][:3], 1):
                print(f"  {i}. {rule['rule']}")
        if result["rules"].get("conditional_rules"):
            print("\nTop 3 Conditional Rules:")
            for i, rule in enumerate(result["rules"]["conditional_rules"][:3], 1):
                print(f"  {i}. {rule['rule']}")
        if rules_path:
            print(f"\nAll rules saved to: {rules_path}")
    print(f"Scores saved to: {scores_csv}")
    print(f"Relations graph saved to: {graph_path}")





