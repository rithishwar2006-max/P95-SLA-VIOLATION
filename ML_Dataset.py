import pandas as pd
import numpy as np
import gc
import os

# ---> SET YOUR FOLDER PATH HERE <---
BASE_PATH = r"C:\Users\rithi\Downloads\ml dataset"
BUCKET = 60000

def get_file_list(prefix, max_idx=5):
    files = []
    for i in range(max_idx + 1):
        path = f"{BASE_PATH}\\{prefix}_{i}.csv"
        if os.path.exists(path):
            files.append(path)
    return files

print("=" * 65)
print("PHASE 1: MAP-REDUCE & GRAPH TOPOLOGY")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# 1A. PROCESS CALLGRAPH & EXTRACT TOPOLOGY
# ─────────────────────────────────────────────────────────────
print("\n[1/4] Aggregating CallGraph files...")
cg_files = get_file_list("CallGraph", 5)
all_edges = []

for file_path in cg_files:
    print(f"  -> Processing {os.path.basename(file_path)}...")
    df_list = []
    for chunk in pd.read_csv(file_path, chunksize=1000000, on_bad_lines='skip', low_memory=False):
        chunk['rt'] = pd.to_numeric(chunk['rt'], errors='coerce')
        chunk.dropna(subset=['rt'], inplace=True)
        chunk['time_window'] = (chunk['timestamp'] // BUCKET) * BUCKET
        
        agg_chunk = chunk.groupby(['um','dm','time_window']).agg(
            p95_latency    = ('rt', lambda x: np.percentile(x, 95)),
            mean_latency   = ('rt', 'mean'),
            max_latency    = ('rt', 'max'),
            total_requests = ('traceid', 'count')
        ).reset_index()
        df_list.append(agg_chunk)
        
    file_agg = pd.concat(df_list, ignore_index=True)
    all_edges.append(file_agg)
    del df_list, file_agg; gc.collect()

edges_agg = pd.concat(all_edges, ignore_index=True)
edges_agg = edges_agg.groupby(['um','dm','time_window']).agg(
    p95_latency    = ('p95_latency', 'max'), 
    mean_latency   = ('mean_latency', 'mean'),
    max_latency    = ('max_latency', 'max'),
    total_requests = ('total_requests', 'sum')
).reset_index()

edges_agg['requests_per_sec'] = edges_agg['total_requests'] / 60.0
del all_edges; gc.collect()

# 🚀 ADVANCED GRAPH TOPOLOGY FEATURES
print("  -> Calculating Graph Centrality (Fan-In/Fan-Out)...")
fan_out = edges_agg.groupby('um')['dm'].nunique().reset_index(name='um_fan_out')
fan_in  = edges_agg.groupby('dm')['um'].nunique().reset_index(name='dm_fan_in')
load    = edges_agg.groupby('dm')['total_requests'].sum().reset_index(name='dm_service_load')

edges_agg = edges_agg.merge(fan_out, on='um', how='left')
edges_agg = edges_agg.merge(fan_in, on='dm', how='left')
edges_agg = edges_agg.merge(load, on='dm', how='left')
print(f"✅ CallGraph & Topology Built! Final Edge shape: {edges_agg.shape}")

# ─────────────────────────────────────────────────────────────
# 1B. PROCESS NODE METRICS
# ─────────────────────────────────────────────────────────────
print("\n[2/4] Aggregating Node Metrics...")
node_files = get_file_list("NodeMetricsUpdate", 5)
all_nodes = []

for file_path in node_files:
    print(f"  -> Processing {os.path.basename(file_path)}...")
    df = pd.read_csv(file_path, on_bad_lines='skip')
    if 'cpu_utiliza' in df.columns:
        df.rename(columns={'cpu_utiliza': 'cpu_utilization', 'memory_util': 'memory_utilization'}, inplace=True)
    df['time_window'] = (df['timestamp'] // BUCKET) * BUCKET
    agg_df = df.groupby(['nodeid','time_window']).agg(
        node_cpu = ('cpu_utilization', 'mean'),
        node_mem = ('memory_utilization', 'mean')
    ).reset_index()
    all_nodes.append(agg_df)
    del df, agg_df; gc.collect()

node_agg = pd.concat(all_nodes, ignore_index=True)
node_agg = node_agg.groupby(['nodeid','time_window']).mean().reset_index()
del all_nodes; gc.collect()
print(f"✅ Node Metrics Aggregated! Shape: {node_agg.shape}")

# ─────────────────────────────────────────────────────────────
# 1C. PROCESS MS METRICS (CONTAINER RESOURCE)
# ─────────────────────────────────────────────────────────────
print("\n[3/4] Aggregating MS Metrics (Resource)...")
ms_files = get_file_list("MSMetricsUpdate", 5)
all_ms = []

for file_path in ms_files:
    print(f"  -> Processing {os.path.basename(file_path)}...")
    df = pd.read_csv(file_path, on_bad_lines='skip')
    if 'cpu_utiliza' in df.columns:
        df.rename(columns={'cpu_utiliza': 'cpu_utilization', 'memory_util': 'memory_utilization'}, inplace=True)
    df['time_window'] = (df['timestamp'] // BUCKET) * BUCKET
    agg_df = df.groupby(['msname','nodeid','time_window']).agg(
        ms_cpu = ('cpu_utilization', 'mean'),
        ms_mem = ('memory_utilization', 'mean')
    ).reset_index()
    all_ms.append(agg_df)
    del df, agg_df; gc.collect()

res_agg = pd.concat(all_ms, ignore_index=True)
res_agg = res_agg.groupby(['msname','nodeid','time_window']).mean().reset_index()
del all_ms; gc.collect()
print(f"✅ MS Metrics Aggregated! Shape: {res_agg.shape}")

# ─────────────────────────────────────────────────────────────
# 1D. PROCESS MCR/RT METRICS (TRAFFIC)
# ─────────────────────────────────────────────────────────────
print("\n[4/4] Aggregating MCR/RT Metrics...")
mcr_files = get_file_list("MCRRTUpdate", 5)
all_mcr = []

for file_path in mcr_files:
    print(f"  -> Processing {os.path.basename(file_path)}...")
    df = pd.read_csv(file_path, on_bad_lines='skip')
    df['time_window'] = (df['timestamp'] // BUCKET) * BUCKET
    agg_df = df.groupby(['msname','time_window']).agg(
        providerrpc_rt = ('providerrpc_rt', 'mean'),
        consumerrpc_rt = ('consumerrpc_rt', 'mean'),
        readdb_rt      = ('readdb_rt', 'mean'),
        writedb_rt     = ('writedb_rt', 'mean'),
        readmc_rt      = ('readmc_rt', 'mean'),
        writemc_rt     = ('writemc_rt', 'mean'),
        http_rt        = ('http_rt', 'mean'),
        read_db_mcr    = ('readdb_mcr', 'mean'),
        write_db_mcr   = ('writedb_mcr', 'mean'),
        http_mcr       = ('http_mcr', 'mean')
    ).reset_index()
    all_mcr.append(agg_df)
    del df, agg_df; gc.collect()

ms_traffic = pd.concat(all_mcr, ignore_index=True)
ms_traffic = ms_traffic.groupby(['msname','time_window']).mean().reset_index()
del all_mcr; gc.collect()
print(f"✅ MCR/RT Aggregated! Shape: {ms_traffic.shape}")

print("\n" + "=" * 65)
print("PHASE 2: MERGING INTO A SINGLE MASTER DATASET")
print("=" * 65)

# Step A: Attach bare-metal node metrics to MS containers
res_with_node = pd.merge(res_agg, node_agg, on=['nodeid','time_window'], how='left')
del res_agg, node_agg; gc.collect()
res_with_node['node_cpu'] = res_with_node['node_cpu'].fillna(-1)
res_with_node['node_mem'] = res_with_node['node_mem'].fillna(-1)

# Step B: Roll up hardware metrics to the MS level
ms_hardware = res_with_node.groupby(['msname','time_window']).agg(
    ms_cpu            = ('ms_cpu', 'mean'),
    ms_mem            = ('ms_mem', 'mean'),
    physical_node_cpu = ('node_cpu', 'mean'),
    physical_node_mem = ('node_mem', 'mean')
).reset_index()
del res_with_node; gc.collect()

# Step C: Combine Hardware + Traffic Profile
master_profile = pd.merge(ms_hardware, ms_traffic, on=['msname','time_window'], how='inner')
del ms_hardware, ms_traffic; gc.collect()

profile_feature_cols = [c for c in master_profile.columns if c not in ['msname','time_window']]
um_rename = {c: f'um_{c}' for c in profile_feature_cols}
dm_rename = {c: f'dm_{c}' for c in profile_feature_cols}

print("\nShrinking RAM usage before massive joins...")
edges_agg['um'] = edges_agg['um'].astype('category')
edges_agg['dm'] = edges_agg['dm'].astype('category')
master_profile['msname'] = master_profile['msname'].astype('category')

# Step D: DUAL JOIN
print("Mapping Topologies (UM -> DM)...")
merged = pd.merge(edges_agg, master_profile, left_on=['um','time_window'], right_on=['msname','time_window'], how='left')
merged.drop(columns=['msname'], inplace=True)
merged.rename(columns=um_rename, inplace=True)

# 🛠️ THE FIX: Safely fill missing metrics with -1 WITHOUT breaking categoricals
fill_dict = {col: -1 for col in merged.columns if col not in ['um', 'dm']}
merged.fillna(value=fill_dict, inplace=True) 

merged = pd.merge(merged, master_profile, left_on=['dm','time_window'], right_on=['msname','time_window'], how='inner')
merged.drop(columns=['msname'], inplace=True)
merged.rename(columns=dm_rename, inplace=True)
del master_profile, edges_agg; gc.collect()

print("\n" + "=" * 65)
print("PHASE 3: TIME-SERIES FEATURE ENGINEERING")
print("=" * 65)

merged.sort_values(by=['um','dm','time_window'], inplace=True)

merged['prev_p95']      = merged.groupby(['um','dm'])['p95_latency'].shift(1)
merged['prev_p95']      = merged['prev_p95'].fillna(merged['p95_latency'])
merged['latency_delta'] = merged['p95_latency'] - merged['prev_p95']

merged['p95_trend_3'] = merged.groupby(['um','dm'])['p95_latency'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
merged['p95_trend_5'] = merged.groupby(['um','dm'])['p95_latency'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

merged['um_cpu_pressure']  = (merged['um_ms_cpu'] > 0.8).astype(int)
merged['dm_cpu_pressure']  = (merged['dm_ms_cpu'] > 0.8).astype(int)
merged['dm_node_pressure'] = (merged['dm_physical_node_cpu'] > 0.8).astype(int)
merged['load_cpu_ratio']   = merged['requests_per_sec'] / (merged['dm_ms_cpu'] + 1e-6)

# ─────────────────────────────────────────────────────────────
# PHASE 4: SLA TARGET (FUTURE PREDICTION)
# ─────────────────────────────────────────────────────────────
SLA_THRESHOLD_MS = merged['p95_latency'].quantile(0.95)
print(f"   Dynamic P95 SLA Threshold: {SLA_THRESHOLD_MS:.2f} ms")

merged['next_window_p95'] = merged.groupby(['um','dm'])['p95_latency'].shift(-1)
merged.dropna(subset=['next_window_p95'], inplace=True)
merged.reset_index(drop=True, inplace=True)

merged['violation_next_window'] = (merged['next_window_p95'] > SLA_THRESHOLD_MS).astype(int)
merged.drop(columns=['next_window_p95'], inplace=True)
merged.fillna(0, inplace=True)

print("\n" + "=" * 65)
print("PHASE 5: FINAL EXPORT")
print("=" * 65)

merged['um'] = merged['um'].astype('category')
merged['dm'] = merged['dm'].astype('category')

OUTPUT_FILE = f"{BASE_PATH}\\Final_Topology_Dataset_FULL_0_to_5__2.csv"
merged.to_csv(OUTPUT_FILE, index=False)

print(f"\nSUCCESS! Fully Graph-Aware Dataset saved to:\n{OUTPUT_FILE}")