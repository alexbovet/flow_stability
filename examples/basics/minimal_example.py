import pandas as pd

from flowstab import FlowStability, set_log_level

set_log_level("INFO")

duration = 600

# URL of the CSV file
url = 'https://zenodo.org/record/4725155/files/mice_contact_sequence.csv.gz'

# Load the CSV file into a DataFrame
mice_contact_df = pd.read_csv(url, compression='gzip')
# limit to the first 60min:
mice_contact_df = mice_contact_df[mice_contact_df['ending_times'] < duration]
# Temporal fix (see #60): reset index
unique_nodes = pd.unique(mice_contact_df[['source_nodes',
                                          'target_nodes']].values.ravel('K'))
node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}
mice_contact_df['source_nodes'] = mice_contact_df['source_nodes'].map(node_mapping)
mice_contact_df['target_nodes'] = mice_contact_df['target_nodes'].map(node_mapping)

# initiate the analysis
fs_mice = FlowStability(
    t_start=None,
    t_stop=None)
print(f"{fs_mice.state.current=}")
print(f"{fs_mice.state.next=}")
# use only events within 10min hour
fs_mice.set_temporal_network(
    events_table=mice_contact_df[mice_contact_df['ending_times'] < duration]
)
print(f"{fs_mice.state.current=}")
print(f"{fs_mice.state.next=}")
fs_mice.compute_laplacian_matrices()
print(f"{fs_mice.state.current=}")
print(f"{fs_mice.state.next=}")

# Set the time scale
fs_mice.time_scale = 1
print(f"{fs_mice.state.current=}")
print(f"{fs_mice.state.next=}")

fs_mice.compute_inter_transition_matrices()
print(f"{fs_mice.state.current=}")
print(f"{fs_mice.state.next=}")

fs_mice.set_flow_clustering()
print(f"{fs_mice.state.current=}")
print(f"{fs_mice.state.next=}")

fs_mice.find_louvain_clustering()
print(f"{fs_mice.state.current=}")
print(f"{fs_mice.state.next=}")
