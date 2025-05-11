import pandas as pd

from flowstab import FlowStability, set_log_level

set_log_level("INFO")

# URL of the CSV file
url = 'https://zenodo.org/record/4725155/files/mice_contact_sequence.csv.gz'

# Load the CSV file into a DataFrame
mice_contact_df = pd.read_csv(url, compression='gzip')
# limit to the first 60min:
mice_contact_df = mice_contact_df[mice_contact_df['ending_times'] < 3600]
# Temporal fix (see #60): reset index
unique_nodes = pd.unique(mice_contact_df[['source_nodes',
                                          'target_nodes']].values.ravel('K'))
node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}
mice_contact_df['source_nodes'] = mice_contact_df['source_nodes'].map(node_mapping)
mice_contact_df['target_nodes'] = mice_contact_df['target_nodes'].map(node_mapping)

# use only events withing 1h hour
fs_mice = FlowStability(
    temporal_network=mice_contact_df[mice_contact_df['ending_times'] < 3600]
)

try:
    fs_mice.compute_laplacian_matrices()
except Exception as e:
    print(e)
# show what needs to be set next
next_steps = fs_mice.next_step()
print(f"Todo next: {next_steps}")
# Explain how
for param in next_steps.get('parameter', {}):
    print(f"###\n{param}\n{fs_mice.howto(param)}\n###\n")

# Set the time scale
fs_mice.time_scale = 1
# Try again
fs_mice.compute_laplacian_matrices()

