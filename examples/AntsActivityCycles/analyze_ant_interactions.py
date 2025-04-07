# For a particular start and end time (t_s, t_e) we want to find a partitioning
# of the tags (i.e. the nodes) that maximises the forward/backward
# (f/b) flow stabilities (Equations (4)&(5) in the paper).
# To do so we need the (f/b) covariance matrices (Equations (2)&(3) in the paper),
# which can be written in terms of a product of inter-event transition matrices (Equation (7)
# in the paper) and the initial probabilities of the nodes to harbor random walkers, i.e.
# p(t_s) and p(t_e).
# For p(t_s) and p(t_e) the assumption of an uniform distribution is made.
# Further, we need to set a resolution parameter, $\lambda$, defining the probability of
# a random walker to move during one unit of time.

# Each of the inter-event transition matrix $\cap(T)(t_k, t_{k+1)$ can be computed as
# $exp(-\lambda L(t_k) \tau_k; \tau_k = t_{k+1} - t_k$ and $L$ the random walk
# graph Laplacian.
# Alternatively one should be able to use the linear interpolation approximation
# (see Equation (14) in the paper).
# 
# Approach:
# - Load the sequence of events into a usable format
# - Define start and stop times, $t_s$ and $t_e$, as well as, the resolution parameter
#   $\lambda$
# - Compute the f/b inter-event transition matrices for the sequence of events to analyze
#   - Compute the laplacian matrices (defined by Equation (9) in the paper)
#   - Compute the inter-event transition matrices
# - Compute the f/b covariance matrices for all pairs $(t_s, t_n), t_n<=t_e$, resp. 
#   $(t_e, t_m), t_m>=t_s$.
# - Find the f/b partitionings that maximise the f/b Flow Stability

import pathlib
import numpy as np
import pandas as pd

from flowstab.TemporalNetwork import ContTempNetwork
from flowstab.FlowStability import FlowIntegralClustering

# ###
# Load the sequence of events into a usable format
# ###
data_path = pathlib.Path("./data/contacts/")
colony_files = data_path.glob("Interactions_Filtered*.txt")

def import_colony_data(source_file: pathlib.Path) -> pd.DataFrame:
    """Load the ant interaction data from the source file
    """
    interaction_df = pd.read_csv(source_file, delimiter=' ',)
    # convert to utc timestamp
    interaction_df['starts'] = pd.to_datetime(interaction_df['Starttime'], unit='s', utc=True)
    interaction_df['stops'] = pd.to_datetime(interaction_df['Stoptime'], unit='s', utc=True)
    return interaction_df

# for c_file in colony_files:
c_file = next(colony_files)  # simply get first to get started
interaction_df = import_colony_data(source_file=c_file)

# NOTE: allow to specify column names in combination with events_table
ctnet = ContTempNetwork(source_nodes=interaction_df.Tag1.to_list(),
                        target_nodes=interaction_df.Tag2.to_list(),
                        starting_times=interaction_df.Starttime.to_list(),
                        ending_times=interaction_df.Stoptime.to_list())

# ###
# Define start and stop times, $t_s$ and $t_e$, as well as, the resolution parameter
# ###
t_s = ctnet.start_time  # = np.min(ctnet.events_table.starting_times)
t_e = ctnet.end_time  # = np.max(ctnet.events_table.ending_times)
# NOTE: This is 10 days of data, too much for a single process>
t_e = t_s + 60. * 60. * 24.  # stop after 1 day
# NOTE: Unsure how lambda is handled since None is valid value
#       > is set to 1/median of event durations if not declared (from docstring of
#       > compute_inter_transition_matrices
# NOTE: rename `lamba` to `l` or `lbd` to avoid confusion with built-in lambda
lbd = None


# ###
# Compute the f/b inter-event transition matrices for the sequence of events to analyze
# ###
# ### ###
#     Compute the laplacian matrices (defined by Equation (9) in the paper)
# ### ###
# NOTE: rename `t_stop` to `t_end` for consistency
ctnet.compute_laplacian_matrices(t_start=t_s, t_stop=t_e, verbose=True)
# NOTE: this takes quite a while
# NOTE: what about splitting the loop from line 1239 to one loop creating the matrices
#       and a 2nd one running in parallel performing the laplacian calculation
# NOTE: ContTempNetwork and ContTempInstNetwork should inherit from same class
#       defining e.g. compute_laplacian_matrices
# ### ###
#     Compute the inter-event transition matrices
# ### ###
# NOTE: rename `t_stop` to `t_end` for consistency
ctnet.compute_inter_transition_matrices(t_start=t_s, t_stop=t_e, lamda=lbd, verbose=True)
# NOTE: this takes forever :P
# NOTE: move check for type of laplacian calculation out of the loop
# NOTE: we could simply run this with multiprocessing (?)
#       CORRECTION: This actually already uses multiple cores!

# ###
# saving
# ###
ctnet.save(filename='ants_ctnet.pickle')
# NOTE: this crashes my 16G RAM laptop (no swap)
# NOTE: $\tau_k = t_{k+1} - t_k$ is subject to the precision of the timestamps
#       of event which might be considerably (and potentially absurdly) high compared
#       to the typical duration of an event.
#       If we were to discretize the raw timestamps we would end up with considerably
#       fewer unique values in the transition matrices and one could think of replacing
#       the actual matrix with a mapping to an array of values when storing them (i.e.
#       the matrices could be stored as np.uint16/32 and save quite some space 

# ###
# Compute the f/b covariance matrices for all pairs $(t_s, t_n), t_n<=t_e$, resp. 
# ###
# NOTE: We need the value of lambda to access the inter event transition matrices
lbd = 1/np.median(np.diff(ctnet.times))
# NOTE: adding a get_lambda method such that lambda=None can also be provided 
#       for ctnet.inter_T[lambda]. Otherwise, if someone never specifies lambda
#       explicitly it needs to be guessed that 1/np.median(np.diff(self.times))
#       is the default value
# NOTE: Why is this in another class?
ficlustering = FlowIntegralClustering(T_inter_list=ctnet.inter_T[lbd],
                                      time_list=ctnet.times[ctnet._k_start_laplacians:ctnet._k_stop_laplacians+1],
                                      verbose=True
                                      )
# NOTE: It seems easier if this were just another method of ctnet
# NOTE: Unclear why there is `T_inter_list` and `T_list` as input arguments
#       > T_list gives the product of the t_inter_list elements from t_1 to t_{k+1}
# NOTE: Not sure if reverse_time can just be set to True to get the bw flow stability.
#       In principle, for un-directional events the inter-event transition matrices
#       should be the agnostic to the direction of time. Would this mean that we
#       can compute the backward flow stability simply by providing reverse_time = True
#       If not, the question is: Why does this attribute exist?
# NOTE: It seems that __init__ directly computes the integral over the covariance matrix.
# NOTE: When computing PT_list, why is it a list (each element is the integral up to t_k?)
# NOTE: How come that ficlustering.I_list then only contains a single element?
# ###
# Find the f/b partitionings that maximise the f/b Flow Stability
# ###
# NOTE: At this point the clustered covariance matrix comes into play (Equation (12) in the
#       paper).
#       To me it is unclear why this is happens via a call to ficlustering.find_louvain_clustering
# NOTE: Also from the paper, for me it is unclear how the Louvain algorithm enters into play here.
