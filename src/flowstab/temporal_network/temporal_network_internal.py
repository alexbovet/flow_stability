import numpy as np
import pandas as pd

class _TemporalNetwork:
    """Continuous time temporal network

    This class represents a continuous time temporal network, where events occur
    between source and target nodes at specified starting and ending times. It
    allows for the management of events, including the option to relabel nodes,
    merge overlapping events, and store additional attributes.
    
    Attributes
    ----------
    events_table : pd.DataFrame
        DataFrame containing event data with columns 'source_nodes',
        'target_nodes', 'starting_times', and 'ending_times'.
    nodes: list
        A list of unique nodes in the network.
    num_nodes : int
        The total number of unique nodes in the network.
    num_events : int
        The total number of events in the network.
    start_time : float
        The minimum starting time of all events.
    end_time : float
        The maximum ending time of all events.
    _compute_times : dict
        A dictionary to record computation times for various operations.
    _overlapping_events_merged : bool
        A flag indicating whether overlapping events have been merged.
    is_directed : bool
        A flag indicating whether the network is directed.
    instantaneous_events : bool
        A flag indicating whether events are instantaneous.
    """
    # parametrize the column names > single place to change them:
    _SOURCES = "source_nodes"
    _TARGETS = "target_nodes"
    _STARTS = "starting_times"
    _ENDINGS = "ending_times"
    _MANDATORY = [_SOURCES, _TARGETS, _STARTS]
    _ESSENTIAL = [_SOURCES, _TARGETS, _STARTS, _ENDINGS]
    # to hold endings - starts
    _DURATIONS = "durations"
    # for instantaneous event this is the duration to use
    _DEFAULT_DURATION = 1

    def __init__(self,
                 events_table:pd.DataFrame|pd.Series|None=None,
                 ):
        """
        Initializes the ContTempNetwork with the given event data.

        Parameters
        ----------
        source_nodes : list
            List of event source nodes, ordered according to `starting_times`.
        target_nodes : list
            List of target nodes for each event.
        starting_times : list
            List of starting times for each event.
        ending_times : list
            List of ending times for each event.
        extra_attrs : dict | None
            Additional event attributes as a dictionary with
            {attr_name: list_of_values}, where list_of_values has the same order
            and length as `source_nodes`.
        merge_overlapping_events : bool, optional
            If True, check for overlapping events (between the same pair of
            nodes) and merge them. Default is False.
        events_table : pd.DataFrame | None
            DataFrame with event data. If provided, it will be used to
            initialize the network instead of the other parameters.
        use_as_is : bool  (Default: `False`)
            If `True`, the provided `events_table` is used without any
            conversion of nodes labels or timestamps.
            By default, `use_as_is` is set to `False` in which case the data
            provided by events_table is copied into an internal data frame
            using standardized indices for nodes.

            Setting `use_as_is` to `True` can lower the memory footprint and
            speedup the initiation of an instance.
            However, one needs to make sure that node labels are consistent,
            i.e. starting with index 0 and ending with $`N-1`$ with $`N`$ being
            the number of nodes.
            This argument is ignored if events are provided via `source_nodes`,
            `target_nodes` and `starting_times`.

        Raises
        ------
        AssertionError
            If the lengths of `source_nodes`, `target_nodes`, `starting_times`,
            and `ending_times` do not match when `events_table` is `None`.
        """ 
        self.events_table = events_table

        # events table is used as is, we still need to extract attributes
        self.nodes = np.sort(pd.unique(self.events_table[
            [self._SOURCES, self._TARGETS]
        ].values.ravel("K"))).tolist()


        # setting some helpful attributes
        self.num_nodes = len(self.nodes)
        self.columns = self.events_table.columns
        self.num_events = self.events_table.shape[0]
        self.start_time = self.events_table[self._STARTS].min()

        # check if the events data frame holds instantaneous events
        if self._ENDINGS not in self.columns:
            self.instantaneous_events = True
            self.end_time = self.events_table[self._STARTS].min()
        else:
            self.events_table[
                self._DURATIONS] = self.events_table[self._ENDINGS] - \
                                   self.events_table[self._STARTS]

        # to record compute times
        self._compute_times = {}

        self._overlapping_events_merged = False
        if merge_overlapping_events:
            num_merged = 1
            while num_merged != 0:
                num_merged = self._merge_overlapping_events()
            self._overlapping_events_merged = True


        self.is_directed = False

    def _get_node(self, node_id):
        """Callable to map internal node IDs back to the provided labels
        """
        return self.nodes[node_id]

    def _get_value(self, value):
        """Callable acting as neutral operator to use for non-converted columns
        """
        return value

    @property
    def events_table(self)->pd.DataFrame:
        """Get the used data in the form of an event table.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the event table data.

        """
        if self._use_as_is:
            return self.events_table
        else:
            # we use self._mappings to retrieve the initially provided values
            # and fall back to just returning the columns as is
            return pd.DataFrame({
                col: self.events_table[col].map(
                    self._mappings.get(col, lambda x: x)
                ) for col in self.columns
            })


    def save(self, filename,
             matrices_list=None,
             attributes_list=None):
        """Save graph events_table and matrices as pickle file
        
        Parameters
        ----------
        filename: str
            Filename where to save. The extension is automatically added.

        matrices_list: list of strings
            List of names of matrices to save.
            The default list is:
                `matrices_list = ['laplacians',
                                  'adjacencies',
                                  'inter_T',
                                  'inter_T_lin',
                                  'T',
                                  'T_lin',
                                  '_stationary_trans',
                                  'delta_inter_T',
                                  'delta_inter_T_lin',]`
        attributes_list: list of strings
            List of attribute names to save.
            The default list is:
                `attributes_list = ['events_table',
                                    '_use_as_is'
                                    'times',
                                    'time_grid',
                                    '_compute_times',
                                    '_t_start_laplacians',
                                    '_k_start_laplacians',
                                    '_t_stop_laplacians',
                                    '_k_stop_laplacians',
                                    '_overlapping_events_merged',]`
                                    
            
        """
        save_dict = dict()


        matrices = ["laplacians",
                    "adjacencies",
                    "inter_T",
                    "inter_T_lin",
                    "T",
                    "T_lin",
                    "_stationary_trans",
                    "delta_inter_T",
                    "delta_inter_T_lin"]

        if matrices_list is None:
            matrices_list = matrices

        attributes = ["nodes",
                      "events_table",
                      "_use_as_is",
                      "times",
                      "time_grid",
                      "_compute_times",
                      "_t_start_laplacians",
                      "_k_start_laplacians",
                      "_t_stop_laplacians",
                      "_k_stop_laplacians",
                      "_overlapping_events_merged",
                      "is_directed"]

        if attributes_list is None:
            attributes_list = attributes

        for attr in attributes_list:
            if hasattr(self, attr):
                save_dict[attr] = getattr(self,attr)


        for mat in matrices_list:
            if hasattr(self, mat):
                save_dict[mat] = getattr(self,mat)


        with open(os.path.splitext(filename)[0] + ".pickle", "wb") as fopen:
            pickle.dump(save_dict, fopen)


    @classmethod
    def load(cls, filename,
             matrices_list=None,
             attributes_list=None):
        """Load network from file and returns a ContTempNetwork instance.
        
        Parameters
        ----------
        filename: str
            Filename where the network is saved save.
            The extension is automatically added.

        matrices_list: list of strings
            List of names of matrices to load.
            The default list is:
                `matrices_list = ['laplacians',
                                    'adjacencies',
                                    'inter_T',
                                    'inter_T_lin',
                                    'T',
                                    'T_lin',
                                    '_stationary_trans',
                                    'delta_inter_T',
                                    'delta_inter_T_lin',]`
        attributes_list: list of strings
            List of attribute names to load.
            The default list is:
                `attributes_list = ['events_table',
                                    '_use_as_is',
                                    'times',
                                    'time_grid',
                                    '_compute_times',
                                    '_t_start_laplacians',
                                    '_k_start_laplacians',
                                    '_t_stop_laplacians',
                                    '_k_stop_laplacians',
                                    '_overlapping_events_merged',]`
        """
        matrices = ["laplacians",
                    "adjacencies",
                    "inter_T",
                    "inter_T_lin",
                    "T",
                    "T_lin",
                    "_stationary_trans",
                    "delta_inter_T",
                    "delta_inter_T_lin"]

        if matrices_list is None:
            matrices_list = matrices

        attributes = ["events_table",
                      "_use_as_is",
                      "times",
                      "time_grid",
                      "_compute_times",
                      "_t_start_laplacians",
                      "_k_start_laplacians",
                      "_t_stop_laplacians",
                      "_k_stop_laplacians",
                      "_overlapping_events_merged",
                      "is_directed"]

        if attributes_list is None:
            attributes_list = attributes


        # all in a pickle file

        graph_dict = pd.read_pickle(os.path.splitext(filename)[0] +".pickle")

        events_table = graph_dict.pop("events_table")

        # we load an exported instance, so it should be save to import the
        # data frame as is (note: below we will overwrite the use as it with
        # the saved state)
        net = cls(events_table=events_table, use_as_is=True)

        # load additional pre-computed attributes
        for k, val in graph_dict.items():
            if k in matrices_list:
                setattr(net, k, val)
            if k in attributes_list:
                setattr(net, k, val)

        return net


    def save_inter_T(self, filename, lamda=None, round_zeros=True, tol=1e-8,
                     compressed=False, save_delta=False,
                     replace_existing=False):
        """ 
        Saves all the inter transition matrices in `self.inter_T[lamda]` 
        in a pickle file together  with a dictionary including parameters:
        `_k_start_laplacians`, `_k_stop_laplacians`, `_t_start_laplacians`,
        `_t_stop_laplacians`, `_t_stop_laplacians`, `times_k_start_to_k_stop+1` 
        (= self.times.values[self._k_start_laplacians:self._k_stop_laplacians+1])
        `num_nodes` and `_compute_times`.

        if `save_delta` is True, creates delta_inter_T if it is
        not already present and saves it together with
        inter_T[lamda][0] in a pickle file.
        otherwise, saves inter_T[lamda] directly (good if used with
        SparseStochMat instances).
            
        Parameters
        ----------
            filename: str
            Filename where the data is saved (.pickle or .gz).
        lamda: float or int, optional.
            Use to save to results for a specific lamba. If None, the results
            for all the computed lambdas will be saved. Default is None.
        round_zeros: bool.
            If True, values smaller than tol*max(abs(inter_T_k)) will be set to
            zero to preserve sparsity. Default is True.
        tol: float
            See round_zeros. Default is 1e-8.
        compressed: bool
            Used to compress the file with gzip. Default is False.
        save_delta: bool
            If True, creates delta_inter_T if it is 
            not already present and saves it together with 
            inter_T[lamda][0]. Only the differences between two consecutives
            inter_Ts are saved in order to minimize file size. 
            Must not be used if `use_sparse_stoch` was used in
            `compute_inter_transition_matrices`.
        replace_existing: bool
            If True, erase and replace files if they already exists. 
            Default is False.
            
        Returns
        -------
            None
        """
        assert hasattr(self, "inter_T"), f"PID {os.getpid()} : nothing saved, "\
            "compute inter_T first."


        ext = os.path.splitext(filename)[-1]

        file = filename

        if compressed:
            if ext != ".gz":
                file += ".gz"
        elif ext != ".pickle":
            file += ".pickle"

        skipping = False
        if os.path.exists(file):
            if replace_existing:
                print(f"PID {os.getpid()}: file {file} already exists, "
                      "replacing it.")
            else:
                print(f"PID {os.getpid()}: file {file} already exists, "
                      "skipping.")
                skipping = True

        if not skipping:


            save_dict = {}
            save_dict["_k_start_laplacians"] = self._k_start_laplacians
            save_dict["_k_stop_laplacians"] = self._k_stop_laplacians
            save_dict["_t_start_laplacians"] = self._t_start_laplacians
            save_dict["_t_stop_laplacians"] = self._t_stop_laplacians
            save_dict["times_k_start_to_k_stop+1"] = self.times.values[
                self._k_start_laplacians:self._k_stop_laplacians+1
            ]
            save_dict["_compute_times"] = self._compute_times

            if save_delta:
                assert not isinstance(self.inter_T[lamda][0], SparseStochMat),\
                                "inter_T must not be SparseStochMat"

                if not hasattr(self, "delta_inter_T") and \
                                    hasattr(self, "inter_T"):
                    if lamda is not None:
                        self._compute_delta_trans_mat(lamda,
                                                      round_zeros=round_zeros,
                                                      tol=tol)
                    else:
                        #computes it for all lamda
                        for lamda in self.inter_T.keys():
                            self._compute_delta_trans_mat(
                                lamda,
                                round_zeros=round_zeros,
                                tol=tol
                            )

                if (lamda is not None) and \
                   (lamda not in self.delta_inter_T.keys()):
                    self._compute_delta_trans_mat(lamda,
                                                  round_zeros=round_zeros,
                                                  tol=tol)

                if hasattr(self, "delta_inter_T"):
                    save_dict["inter_T"] = dict()
                    save_dict["is_delta_trans"] = True

                    if lamda is None:
                        lamdas = self.delta_inter_T.keys()
                    else:
                        lamdas = [lamda]

                    for lamda in lamdas:
                        save_dict["inter_T"][lamda] = dict()
                        save_dict["inter_T"][lamda]["delta_inter_T"] = \
                            self.delta_inter_T[lamda]
                        save_dict["inter_T"][lamda]["trans_mat0"] = \
                            self.inter_T[lamda][0].copy()
                        if round_zeros:
                            set_to_zeroes(
                                save_dict["inter_T"][lamda]["trans_mat0"],
                                tol=tol
                            )

                text = "delta trans mats"

            else:

                save_dict["inter_T"] = dict()
                save_dict["is_sparse_stoch"] = True

                if lamda is None:
                    lamdas = self.inter_T.keys()
                else:
                    lamdas = [lamda]

                for lamda in lamdas:
                    assert isinstance(self.inter_T[lamda][0],
                                      SparseStochMat),\
                                    "inter_T needs to be SparseStochMat"

                    save_dict["inter_T"][lamda] = []
                    for interT in self.inter_T[lamda]:
                        if round_zeros:
                            interT.set_to_zeroes(tol)
                        save_dict["inter_T"][lamda].append(interT.to_dict())

                text = "sparse stoch trans mats"


            if compressed:

                print(f"PID {os.getpid()}: saving {text} to '{file}'")

                with gzip.open(file,
                               "wb", compresslevel=2) as fopen:
                    pickle.dump(save_dict, fopen)
            else:

                print(f"PID {os.getpid()}: saving {text} to '{file}'")

                with open(file, "wb") as fopen:
                    pickle.dump(save_dict, fopen)


    def save_inter_T_lin(self,
                         filename,
                         lamda=None,
                         round_zeros=True,
                         tol=1e-8,
                         compressed=False,
                         replace_existing=True,
                         save_delta=False):
        """Creates delta_inter_T_lin if it is not already present and
        saves it together with inter_T_lin[lamda][t_s][0] in a pickle file.
                        
        """
        assert hasattr(self, "inter_T_lin"),\
            f"PID {os.getpid()} : nothing saved, compute inter_T_lin first."

        ext = os.path.splitext(filename)[-1]

        file = filename

        if compressed:
            if ext != ".gz":
                file += ".gz"
        elif ext != ".pickle":
            file += ".pickle"

        skipping = False
        if os.path.exists(file):
            if replace_existing:
                print(f"PID {os.getpid()}: , file {file} already exists, "
                    "replacing it.")
            else:
                print(f"PID {os.getpid()}: , file {file} already exists, "
                    "skipping.")
                skipping = True

        if not skipping:

            save_dict = {}
            save_dict["_k_start_laplacians"] = self._k_start_laplacians
            save_dict["_k_stop_laplacians"] = self._k_stop_laplacians
            save_dict["_t_start_laplacians"] = self._t_start_laplacians
            save_dict["_t_stop_laplacians"] = self._t_stop_laplacians
            save_dict["times_k_start_to_k_stop+1"] = self.times.values[
                self._k_start_laplacians:self._k_stop_laplacians+1
            ]
            save_dict["num_nodes"] = self.num_nodes
            save_dict["_compute_times"] = self._compute_times

            if save_delta:
                if not hasattr(self, "delta_inter_T_lin") and \
                                    hasattr(self, "inter_T_lin"):
                    if lamda is not None:
                        self._compute_delta_trans_mat(lamda,
                                                      round_zeros=round_zeros,
                                                      tol=tol)
                    else:
                        #computes it for all lamda
                        for lamda in self.inter_T_lin.keys():
                            self._compute_delta_trans_mat(
                                lamda,
                                round_zeros=round_zeros,
                                tol=tol
                            )

                if (lamda is not None) and\
                    (lamda not in self.delta_inter_T_lin.keys()):
                    self._compute_delta_trans_mat(lamda,
                                                  round_zeros=round_zeros,
                                                  tol=tol)

                if hasattr(self, "delta_inter_T_lin"):


                    save_dict["inter_T_lin"] = dict()
                    save_dict["is_delta_trans"] = True

                    if lamda is None:
                        lamdas = self.delta_inter_T_lin.keys()
                    else:
                        lamdas = [lamda]

                    for lamda in lamdas:
                        save_dict["inter_T_lin"][lamda] = dict()
                        for t_s in self.inter_T_lin[lamda].keys():
                            save_dict["inter_T_lin"][lamda][t_s] = dict()
                            save_dict["inter_T_lin"][lamda][t_s][
                                "delta_inter_T_lin"
                            ] = self.delta_inter_T_lin[lamda][t_s]
                            save_dict["inter_T_lin"][lamda][t_s][
                                "trans_mat_lin0"
                            ] = self.inter_T_lin[lamda][t_s][0].copy()
                            if round_zeros:
                                set_to_zeroes(
                                    save_dict["inter_T_lin"][lamda][t_s][
                                        "trans_mat_lin0"
                                    ],
                                    tol=tol
                                )

                text = "delta trans mats"

            else:
                save_dict["inter_T_lin"] = dict()
                save_dict["is_sparse_stoch"] = True

                if lamda is None:
                    lamdas = self.delta_inter_T_lin.keys()
                else:
                    lamdas = [lamda]

                for lamda in lamdas:
                    save_dict["inter_T_lin"][lamda] = dict()
                    for t_s in self.inter_T_lin[lamda].keys():
                        assert isinstance(self.inter_T_lin[lamda][t_s][0],
                                          SparseStochMat),\
                                        "inter_T needs to be SparseStochMat"

                        save_dict["inter_T_lin"][lamda][t_s] = []
                        for interT in self.inter_T_lin[lamda][t_s]:
                            if round_zeros:
                                interT.set_to_zeroes(tol=tol)
                            save_dict["inter_T_lin"][lamda][t_s].append(
                                interT.to_dict()
                            )

                text = "sparse stoch trans mats"



            if compressed:

                print(f"PID {os.getpid()}: saving '{text}' to '{file}'")

                with gzip.open(file,
                               "wb", compresslevel=2) as fopen:
                    pickle.dump(save_dict, fopen)
            else:

                print(f"PID {os.getpid()}: saving '{text}' to '{file}'")

                with open(file, "wb") as fopen:
                    pickle.dump(save_dict, fopen)


    @staticmethod
    def load_inter_T(filename):
        """Loads inter_T and inter_T_lin from 'filename' that was saved with
        save_inter_T.
            
        returns a dictionary with the inter_T restored.
            
        """
        ext = os.path.splitext(filename)[-1]


        if ext not in [".pickle",".gz"]:
            # detect extension
            if os.path.exists(filename + ".pickle"):
                ext = ".pickle"
                filename += ".pickle"
            elif os.path.exists(filename + ".gz"):
                ext = ".gz"
                filename += ".gz"
            elif os.path.exists(filename + ".pickle.gz"):
                ext = ".pickle.gz"
                filename += ".pickle.gz"
            else:
                raise FileNotFoundError(filename)

        if ext == ".gz" or ext == ".pickle.gz":
            with gzip.open(filename,
                           "rb") as fopen:
                load_dict = pickle.load(fopen)

        else:
            with open(filename, "rb") as fopen:
                load_dict = pickle.load(fopen)

        return_dict = {
            "_k_start_laplacians" : load_dict["_k_start_laplacians"],
            "_k_stop_laplacians" : load_dict["_k_stop_laplacians"],
            "_t_start_laplacians" : load_dict["_t_start_laplacians"],
            "_t_stop_laplacians" : load_dict["_t_stop_laplacians"],
            "num_nodes" : load_dict["num_nodes"],
            "times_k_start_to_k_stop+1" : load_dict["times_k_start_to_k_stop+1"]
        }

        # rebuild inter_T from delta_inter_T
        if "inter_T" in load_dict.keys():
            return_dict["inter_T"] = dict()

            if load_dict.get("is_sparse_stoch", False):

                for lamda in load_dict["inter_T"].keys():
                    return_dict["inter_T"][lamda] = \
                        [SparseStochMat(**mat_dict) for mat_dict in \
                                          load_dict["inter_T"][lamda]]

            else:
                for lamda in load_dict["inter_T"].keys():
                    return_dict["inter_T"][lamda] = \
                        [load_dict["inter_T"][lamda]["trans_mat0"]]

                    for dT in load_dict["inter_T"][lamda]["delta_inter_T"]:
                        return_dict["inter_T"][lamda].append(\
                                 return_dict["inter_T"][lamda][-1] + dT)


        if "inter_T_lin" in load_dict.keys():
            return_dict["inter_T_lin"] = dict()

            if load_dict.get("is_sparse_stoch", False):

                for lamda in load_dict["inter_T_lin"].keys():
                    return_dict["inter_T_lin"][lamda] = dict()

                    for t_s in load_dict["inter_T_lin"][lamda].keys():

                        return_dict["inter_T_lin"][lamda][t_s] = [
                            SparseStochMat(**mat_dict)
                            for mat_dict in load_dict["inter_T_lin"][lamda][t_s]
                        ]

            else:
                for lamda in load_dict["inter_T_lin"].keys():
                    return_dict["inter_T_lin"][lamda] = dict()

                    for t_s in load_dict["inter_T_lin"][lamda].keys():

                        return_dict["inter_T_lin"][lamda][t_s] = [load_dict[
                            "inter_T_lin"
                        ][lamda][t_s]["trans_mat_lin0"]]

                        for dT in load_dict["inter_T_lin"][lamda][t_s][
                            "delta_inter_T_lin"
                        ]:
                            return_dict["inter_T_lin"][lamda][t_s].append(
                                return_dict["inter_T_lin"][lamda][t_s][-1]+dT
                            )

        del load_dict
        return return_dict

    def save_T(self, filename, lamda=None, round_zeros=True, tol=1e-8,
               compressed=False):
        """ 
        saves a dict with 'T' as key and net.T as item with other 
        useful attributes.
            
        also works with SparseStochMat.
            
        only works if net.T[lamda] is a matrix and not a list of matrices,
        i.e. if compute_transition_matrices was ran without save_intermediate.
                        
        """
        assert hasattr(self, "T"), \
            f"PID {os.getpid()} : nothing saved, compute inter_T first."

        save_dict = {}
        save_dict["_k_start_laplacians"] = self._k_start_laplacians
        save_dict["_k_stop_laplacians"] = self._k_stop_laplacians
        save_dict["_t_start_laplacians"] = self._t_start_laplacians
        save_dict["_t_stop_laplacians"] = self._t_stop_laplacians
        save_dict["times_k_start_to_k_stop+1"] = self.times.values[
            self._k_start_laplacians:self._k_stop_laplacians+1
        ]
        save_dict["num_nodes"] = self.num_nodes
        save_dict["_compute_times"] = self._compute_times

        save_dict["T"] = dict()

        if lamda is None:
            lamdas = self.T.keys()
        else:
            lamdas = [lamda]

        for lamda in lamdas:
            if isinstance(self.T[lamda], SparseStochMat):
                save_dict["is_sparse_stoch"] = True
                if round_zeros:
                    T = self.T[lamda].copy()
                    T.set_to_zeroes(tol)
                else:
                    T = self.T[lamda]
                save_dict["T"][lamda] = T.to_dict()

                text = "SparseStochMat T"

            elif isspmatrix_csr(self.T[lamda]):

                if round_zeros:
                    T = self.T[lamda].copy()
                    set_to_zeroes(T, tol)
                else:
                    T = self.T[lamda]
                save_dict["T"][lamda] = T

                text = "csr T"

            else:
                raise TypeError("T must be csr or SparseStochMat.")


        ext = os.path.splitext(filename)[-1]

        file = filename

        if compressed:

            if ext != ".gz":
                file += ".gz"

            print(f"PID {os.getpid()}: saving '{text}' to '{file}'")

            with gzip.open(file,
                           "wb", compresslevel=2) as fopen:
                pickle.dump(save_dict, fopen)
        else:
            ext = os.path.splitext(filename)[-1]
            if ext != ".pickle":
                file += ".pickle"

            print(f"PID {os.getpid()}: saving '{text}' to '{file}'")

            with open(file, "wb") as fopen:
                pickle.dump(save_dict, fopen)

    def save_T_lin(self, filename, lamda=None, round_zeros=True, tol=1e-8,
                   compressed=False):
        """ 
        saves a dict with 'T_lin' as key and net.T_lin as item with other 
        useful attributes.
            
        also works with SparseStochMat instance.
            
        `net.T_lin[lamda][t_s]` must be a matrix and not a list of matrices,
        i.e. if compute_transition_matrices was ran without save_intermediate.
                        
        """
        assert hasattr(self, "T_lin"), \
            f"PID {os.getpid()} : nothing saved, compute inter_T_lin first."

        save_dict = {}
        save_dict["_k_start_laplacians"] = self._k_start_laplacians
        save_dict["_k_stop_laplacians"] = self._k_stop_laplacians
        save_dict["_t_start_laplacians"] = self._t_start_laplacians
        save_dict["_t_stop_laplacians"] = self._t_stop_laplacians
        save_dict["times_k_start_to_k_stop+1"] = self.times.values[
            self._k_start_laplacians:self._k_stop_laplacians+1
        ]
        save_dict["num_nodes"] = self.num_nodes
        save_dict["_compute_times"] = self._compute_times

        save_dict["T_lin"] = dict()

        if lamda is None:
            lamdas = self.T_lin.keys()
        else:
            lamdas = [lamda]

        for lamda in lamdas:
            save_dict["T_lin"][lamda] = dict()
            for t_s in self.T_lin[lamda].keys():

                if isinstance(self.T_lin[lamda][t_s], SparseStochMat):
                    save_dict["is_sparse_stoch"] = True
                    if round_zeros:
                        T = self.T_lin[lamda][t_s].copy()
                        T.set_to_zeroes(tol)
                    else:
                        T = self.T_lin[lamda][t_s]

                    save_dict["T_lin"][lamda][t_s] = T.to_dict()

                    text = "SparseStochMat T_lin"

                elif isspmatrix_csr(self.T_lin[lamda][t_s]):

                    if round_zeros:
                        T = self.T_lin[lamda][t_s].copy()
                        set_to_zeroes(T, tol)
                    else:
                        T = self.T_lin[lamda][t_s]
                    save_dict["T_lin"][lamda][t_s] = T

                    text = "csr T_lin"

                else:
                    raise TypeError("T must be csr or SparseStochMat.")


        ext = os.path.splitext(filename)[-1]

        file = filename

        if compressed:

            if ext != ".gz":
                file += ".gz"

            print(f"PID {os.getpid()}: saving '{text}' to '{file}'")

            with gzip.open(file,
                           "wb", compresslevel=2) as fopen:
                pickle.dump(save_dict, fopen)
        else:
            ext = os.path.splitext(filename)[-1]
            if ext != ".pickle":
                file += ".pickle"

            print(f"PID {os.getpid()}: saving '{text}' to '{file}'")

            with open(file, "wb") as fopen:
                pickle.dump(save_dict, fopen)

    @staticmethod
    def load_T(filename):
        """Loads T and T_lin from 'filename' that was saved with save_T.
            
        returns a dictionary with the T restored.
            
        """
        ext = os.path.splitext(filename)[-1]


        if ext not in [".pickle",".gz"]:
            # detect extension
            if os.path.exists(filename + ".pickle"):
                ext = ".pickle"
                filename += ".pickle"
            elif os.path.exists(filename + ".gz"):
                ext = ".gz"
                filename += ".gz"
            elif os.path.exists(filename + ".pickle.gz"):
                ext = ".pickle.gz"
                filename += ".pickle.gz"
            else:
                raise FileNotFoundError(filename)

        if ext == ".gz" or ext == ".pickle.gz":
            with gzip.open(filename,
                           "rb") as fopen:
                load_dict = pickle.load(fopen)

        else:
            with open(filename, "rb") as fopen:
                load_dict = pickle.load(fopen)

        return_dict = {
            "_k_start_laplacians" : load_dict["_k_start_laplacians"],
            "_k_stop_laplacians" : load_dict["_k_stop_laplacians"],
            "_t_start_laplacians" : load_dict["_t_start_laplacians"],
            "_t_stop_laplacians" : load_dict["_t_stop_laplacians"],
            "num_nodes" : load_dict["num_nodes"],
            "times_k_start_to_k_stop+1" : load_dict["times_k_start_to_k_stop+1"]
        }


        if "T" in load_dict.keys():
            return_dict["T"] = dict()

            if load_dict.get("is_sparse_stoch", False):

                for lamda in load_dict["T"].keys():
                    return_dict["T"][lamda] = \
                        SparseStochMat(**load_dict["T"][lamda])

            else:
                for lamda in load_dict["T"].keys():
                    return_dict["T"][lamda] = load_dict["T"][lamda]


        if "T_lin" in load_dict.keys():
            return_dict["T_lin"] = dict()

            if load_dict.get("is_sparse_stoch", False):

                for lamda in load_dict["T_lin"].keys():
                    return_dict["T_lin"][lamda] = dict()

                    for t_s in load_dict["T_lin"][lamda].keys():

                        return_dict["T_lin"][lamda][t_s] = \
                            SparseStochMat(**load_dict["T_lin"][lamda][t_s])

            else:
                for lamda in load_dict["T_lin"].keys():
                    return_dict["T_lin"][lamda] = dict()

                    for t_s in load_dict["T_lin"][lamda].keys():

                        return_dict["T_lin"][lamda][t_s] = \
                                            load_dict["T_lin"][lamda][t_s]



        del load_dict

        return return_dict


    def compute_static_adjacency_matrix(self,
                               start_time=None,
                               end_time=None):
        """Returns the adjacency matrix of the static network built from the
        aggregagted edge activity between `start_time` and `end_time`.

        Parameters
        ----------
        start_time : float or int, optional
            Starting time for the aggregation. The default is None, i.e. the
            start time of the entire temporal network.
        end_time : float or int, optional
            Ending time for the aggregation. The default is None, i.e. the
            end time of the entire temporal network.

        Returns
        -------
        CSR sparse matrix
            Symmetric adjacency matrix, where element ij is equal to the
            aggregated time during which egde ij was active after `start_time`
            and before `end_time`.

        """
        if start_time is None:
            start_time = self.start_time

        if end_time is None:
            end_time = self.end_time

        mask = np.logical_and(self.events_table.starting_times < end_time,
                              self.events_table.ending_times > start_time)

        # loop on events
        data = []
        cols = []
        rows = []
        for ev in self.events_table.loc[mask].itertuples():
            data.append(min(ev.ending_times, end_time) - max(ev.starting_times,
                                                             start_time))
            rows.append(ev.source_nodes)
            cols.append(ev.target_nodes)

        A = coo_matrix((data, (rows,cols)),
                       shape=(self.num_nodes, self.num_nodes))

        return A + A.T


    def _compute_time_grid(self):
        """Create `self.time_grid`, a dataframe with ('times', 'id') as index,
        were `id` is the index of the corresponding event in
        `self.events_table`, and column 'is_start' which is True is the
        ('times', 'id') corresponds to a starting event.
        Also creates `self.times`, an array with all the times values.
            
        """
        self.time_grid = pd.DataFrame(
            columns=["times", "id", "is_start"],
            index=range(self.events_table.shape[0]*2)
        )
        self.time_grid.iloc[:self.events_table.shape[0], [0, 1]] = \
            self.events_table.reset_index()[[self._STARTS, "index"]]
        self.time_grid["is_start"] = False
        self.time_grid.loc[0:self.events_table.shape[0] - 1,"is_start"] = True

        self.time_grid.iloc[self.events_table.shape[0]:, [0, 1]] = \
            self.events_table.reset_index()[[self._ENDINGS, "index"]]

        self.time_grid.times = pd.to_numeric(self.time_grid.times)

        self.time_grid.sort_values("times", inplace=True)

        # group events with same times
        self.time_grid.set_index(["times", "id"], inplace=True)

        self.time_grid.sort_index(inplace=True)

        self.times = self.time_grid.index.levels[0]


    def _get_closest_time(self, t):
        """Return closest smaller or equal time in `times` and its index"""
        if not hasattr(self, "times"):
            self._compute_time_grid()

        if t not in self.times:
        # take the largest smaller time
            if t <= self.times[0]:
                t = self.times[0]
            else:
                t = self.times[self.times <= t].max()

        k = int(np.where(self.times == t)[0])

        return t, k


    def compute_laplacian_matrices(self,
                                   t_start=None,
                                   t_stop=None,
                                   verbose=False,
                                   save_adjacencies=False):
        """Computes the laplacian matrices and saves them in `self.laplacians`
            
        Computes from the first event time (in `self.times`) before or equal to
        `t_start` until the event time index before `t_stop`.
        
        Laplacians are computed from `self.times[self._k_start_laplacians]`
        until `self.times[self._k_stop_laplacians-1]`.
        
        The laplacian at step k, is the random walk laplacian
        between `times[k]`. and `times[k+1]`.
            
        Parameters
        ----------
        t_start : float or int, optional
            The default is None, i.e. starts at the beginning of times.
            The computation starts from the first time index before or equal to
            `t_start`.
            The corresponding starting time index is saved in
            `self._k_start_laplacians` and the real starting time is
            `self.times[self._k_start_laplacians]` which is saved in
            `self._t_start_laplacians`.
        t_stop : float or int, optional
            Same than `t_start` but for the ending time of computations.
            Default is end of times.
            Computations stop at self.times[self._k_stop_laplacians-1].
            Similarly to `t_start`, the corresponding times are saved in 
            `self._k_stop_laplacians` and `self._t_stop_laplacians`.
        verbose : bool, optional
            The default is False.
        save_adjacencies : bool, optional
            Default is False. Use to save adjacency matrices in
            `self.adjacencies`.

        Returns
        -------
        None.            

        """
        if verbose:
            print(f"PID {os.getpid()}: Computing Laplacians")

        if not hasattr(self, "time_grid"):
            self._compute_time_grid()

        if self.instantaneous_events:
            # instantaneous adjacency matrix
            A = dok_matrix((self.num_nodes, self.num_nodes),
                           dtype=np.float64)
        else:
            # instantaneous adjacency matrix
            A = lil_matrix((self.num_nodes, self.num_nodes),
                           dtype=np.float64)

        #identity
        I = eye(self.num_nodes,
                dtype=np.float64).tocsc()

        #degree array
        degrees = np.zeros(self.num_nodes, dtype=np.float64)
        # inverse degrees diagonal matrix
        Dm1 = I.copy()
        # self loop matrix
        S = I.copy()


        self.laplacians = []
        if save_adjacencies:
            self.adjacencies = []

        # set boundary conditions : L_k is the laplacian during t_k and t_k+1
        if t_start is None:
            self._t_start_laplacians = self.times[0]
            self._k_start_laplacians = 0
        else:
            t, k = self._get_closest_time(t_start)
            self._t_start_laplacians = t
            self._k_start_laplacians = k
        if t_stop is None:
            self._t_stop_laplacians = self.times[-1]
            self._k_stop_laplacians = len(self.times)-1
        else:
            t, k = self._get_closest_time(t_stop)
            self._t_stop_laplacians = t
            self._k_stop_laplacians = k

        t0 = time.time()


        if not self.instantaneous_events and self._k_start_laplacians > 0:
            # initial conditions, we have to find the adjacency mat just before
            # _k_start_laplacians.

            t_km1 = self.times[self._k_start_laplacians-1]

            # find events that have started before or at t_k-1
            # and were still occuring at t_k-1
            mask_ini = (self.events_table[self._STARTS] <= t_km1) & \
                       (self.events_table[self._ENDINGS] > t_km1)

            for event in self.events_table.loc[mask_ini][[
                self._SOURCES,
                self._TARGETS
            ]].itertuples():

                if A[event.source_nodes, event.target_nodes] != 1:
                        A[event.source_nodes, event.target_nodes] = 1
                        A[event.target_nodes, event.source_nodes] = 1

                        degrees[event.source_nodes] += 1
                        degrees[event.target_nodes] += 1

                if degrees[event.source_nodes] == 0:
                        S.data[event.source_nodes] = 1
                        Dm1.data[event.source_nodes] = 1
                else:
                    S.data[event.source_nodes] = 0
                    Dm1.data[event.source_nodes] = 1/degrees[event.source_nodes]

                if degrees[event.target_nodes] == 0:
                    S.data[event.target_nodes] = 1
                    Dm1.data[event.target_nodes] = 1
                else:
                    S.data[event.target_nodes] = 0
                    Dm1.data[event.target_nodes] = 1/degrees[event.target_nodes]

        # time grid for this time range
        time_grid_range = self.time_grid.loc[
            (self.time_grid.index.get_level_values("times") >= \
                self._t_start_laplacians) & \
            (self.time_grid.index.get_level_values("times") < \
                self._t_stop_laplacians)
        ]

        for k, (tk, time_ev) in enumerate(
            time_grid_range.groupby(level="times")
        ):
            if verbose and not k%1000:
                print(f"PID {os.getpid()}: {k} over " ,
                      self._k_stop_laplacians - self._k_start_laplacians)
                print(f"PID {os.getpid()} : {time.time()-t0:.2f}s")

            meet_id = time_ev.index.get_level_values("id")
            # starting or ending events
            is_starts = time_ev.is_start.values

        # all in

            events_k = [
                self.events_table.loc[
                    mid, [self._SOURCES, self._TARGETS]
                ].astype(np.int64)
                for mid in meet_id.values
            ]

            print(f"{events_k=}")
            # print(f"{self.events_table}")
            # print(f"{A=}")

            #update instantaneous matrices
            for event, is_start in zip(events_k, is_starts):
                # unweighted, undirected
                if is_start:
                    # if they are not already connected (can happen if the
                    # opposite event overlap)
                    if A[event.source_nodes, event.target_nodes] != 1:
                        A[event.source_nodes, event.target_nodes] = 1
                        A[event.target_nodes, event.source_nodes] = 1

                        degrees[event.source_nodes] += 1
                        degrees[event.target_nodes] += 1

                elif A[event.source_nodes, event.target_nodes] > 0:
                    A[event.source_nodes, event.target_nodes] = 0
                    A[event.target_nodes, event.source_nodes] = 0

                    degrees[event.source_nodes] -= 1
                    degrees[event.target_nodes] -= 1

                # update self loops
                if degrees[event.source_nodes] == 0:
                    S.data[event.source_nodes] = 1
                    Dm1.data[event.source_nodes] = 1
                else:
                    S.data[event.source_nodes] = 0
                    Dm1.data[event.source_nodes] = 1/degrees[event.source_nodes]

                if degrees[event.target_nodes] == 0:
                    S.data[event.target_nodes] = 1
                    Dm1.data[event.target_nodes] = 1
                else:
                    S.data[event.target_nodes] = 0
                    Dm1.data[event.target_nodes] = 1/degrees[event.target_nodes]

            # Laplacian L(tk)
            Acsc = A.tocsc()
            # T_D = Dm1 @ (Acsc + S)
            # L = I - T_D

            self.laplacians.append(I - Dm1 @ (Acsc + S))
            if save_adjacencies:
                self.adjacencies.append(A.copy())

        t_end = time.time()-t0
        self._compute_times["laplacians"] = t_end
        if verbose:
            print("PID ", os.getpid(), " : ","finished in ", t_end)

    def compute_inter_transition_matrices(self,
                                          lamda=None,
                                          t_start=None,
                                          t_stop=None,
                                          verbose=False,
                                          use_sparse_stoch=False,
                                          dense_expm=True):
        """Computes interevent transition matrices.

        T_k(lamda) = expm(-tau_k*lamda*L_k).
        
        The transition matrix T_k is saved in `self.inter_T[lamda][k]`, where 
        self.inter_T is a dictionary with lamda as keys and lists of transition
        matrices as values.
        
        will compute from self.times[self._k_start_laplacians] until 
        self.times[self._k_stop_laplacians-1]
        
        the transition matrix at step k, is the probability transition matrix
        between times[k] and times[k+1]
        
        Parameters
        ----------
        lamda : float, optional
            Random walk rate, dynamical resolution parameter. The default (None)
            is 1 over the median inter event time.
        t_start : float or int, optional
            Starting time, passed to `compute_laplacian_matrices` if the 
            Laplacians have not yet been computed.
            Otherwise is not used.
            The computation starts at self.times[self._k_start_laplacians].
            The default is None, i.e. starts at the beginning of times.
        t_stop : float or int, optional
            Same than `t_start` but for the ending time of computations.
            Computations stop at self.times[self._k_stop_laplacians-1].
            Default is end of times.
        verbose : bool, optional
            The default is False.
        use_sparse_stoch : bool, optional
            Whether to use custom sparse stochastic matrix format to save the
            inter transition matrices. Especially useful for large networks as
            the matrix exponential is then computed on each connected component
            separately (more memory efficient). The default is False.
        dense_expm : bool, optional
            Whether to use the dense version of the matrix exponential algorithm
            at each time steps. Recommended for not too large networks. 
            The inter trans. matrices are still saved as sparse scipy matrices
            as they usually have many zero values. The default is True. Has no
            effect is use_sparse_stoch is True.

        Returns
        -------
        None.

        """
        if not hasattr(self, "laplacians"):
            self.compute_laplacian_matrices(t_start=t_start, t_stop=t_stop,
                                            verbose=verbose)
        if not hasattr(self, "inter_T"):
            self.inter_T = dict()


        if lamda is None:
            if verbose:
                print(f"PID {os.getpid()}: taking lamda as 1/tau_w with "
                      "tau_w = median interevent time")
            lamda = 1/np.median(np.diff(self.times))

        # new value of lamda, we need to recompute
        if lamda not in self.inter_T.keys():

            if verbose:
                print(f"PID {os.getpid()}: Computing interevent transition "
                      f"matrices for lamda={lamda}")

            self.inter_T[lamda] = []

            t0 = time.time()
            for k, tk in enumerate(self.times[self._k_start_laplacians:
                                              self._k_stop_laplacians]):
                if verbose and not k%1000:
                    print("PID ", os.getpid(), " : ",k, " over " ,
                          self._k_stop_laplacians-1-self._k_start_laplacians)
                    print(f"PID {os.getpid()} : {time.time()-t0:.2f}s")

                if self.instantaneous_events:
                    tau_k = 1.0
                else:
                    tau_k = self.times[self._k_start_laplacians+k+1] - tk

                if use_sparse_stoch:
                    self.inter_T[lamda].append(
                        sparse_lapl_expm(self.laplacians[k],
                                         tau_k*lamda,
                                         dense_expm=dense_expm)
                    )
                elif self.laplacians[k].getnnz() == 0:
                    # expm of zero = identity
                    self.inter_T[lamda].append(eye(self.num_nodes,format="csr"))
                elif dense_expm:
                    self.inter_T[lamda].append(
                        csr_matrix(
                            expm(-tau_k*lamda*self.laplacians[k].toarray())
                        )
                    )
                else:
                    self.inter_T[lamda].append(
                        expm(-tau_k*lamda*self.laplacians[k]).tocsr()
                    )

            if len(self.inter_T[lamda]) == 0:
                if verbose:
                    print(f"PID {os.getpid()} no events, "
                          "trans. matrix = identity")
                # is there was no event, the transition is identity
                if use_sparse_stoch:
                    self.inter_T[lamda].append(
                        SparseStochMat.create_diag(size=self.num_nodes)
                    )
                else:
                    self.inter_T[lamda].append(eye(self.num_nodes,
                                                      dtype=np.float64,
                                                      format="csr"))

            t_end = time.time()-t0

            self._compute_times["inter_T_" + str(lamda)]  = t_end

            if verbose:
                print(f"PID {os.getpid()}: Finished computing interevent "
                      f"transition matrices in {t_end}")
        elif verbose:
            print(f"PID {os.getpid()}: Interevent transition matrices already "
                  f"computed for lamda={lamda}")

    # TODO: get rid of fix_tau_k in favour of self.instantaneous_events
    def compute_lin_inter_transition_matrices(self,
                                              lamda=None,
                                              t_start=None,
                                              t_stop=None,
                                              verbose=False,
                                              t_s=10,
                                              use_sparse_stoch=False):
        """Compute interevent transition matrices as a linear approximation of
        expm(-tau_k*lamda*L_k) based on the discrete time transition matrix.
            
        `t_s` is the time value for which the linear approximation reaches the
        stationary transition matrix (default is `t_s=10`).
        
        The transition matrix T_k_lin is saved in 
        `self.inter_T_lin[lamda][t_s][k]`,
        where `self.inter_T_lin` is a dictionary with lamda as keys and
        lists of transition matrices as values.
            
        will compute from self.times[self._k_start_laplacians]
        until self.times[self._k_stop_laplacians-1]
            
        the transition matrix at step k, is the probability transition matrix
        between times[k] and times[k+1]
            
        """
        I = eye(self.num_nodes,
                dtype=np.float64, format="csr")

        if not hasattr(self, "_stationary_trans"):
            self._compute_stationary_transition(t_start=t_start, t_stop=t_stop,
                                            verbose=verbose,
                                            use_sparse_stoch=use_sparse_stoch)

        if not hasattr(self, "inter_T_lin"):
            self.inter_T_lin = dict()

        if lamda is None:
            if verbose:
                print(f"PID {os.getpid()}: taking lamda as 1/tau_w with "
                      "tau_w = median interevent time")
            lamda = 1/np.median(np.diff(self.times))

        compute = True

        if lamda in self.inter_T_lin.keys():
            if t_s in self.inter_T_lin[lamda].keys():
                compute = False
                if verbose:
                    print(f"PID {os.getpid()}: Interevent transition matrices "
                          f"already computed for {lamda=}, {t_s=}")

        if compute:
            if lamda not in self.inter_T_lin.keys():
                self.inter_T_lin[lamda] = dict()
                self.inter_T_lin[lamda][t_s] = []
            else:
                self.inter_T_lin[lamda][t_s] = []

            if verbose:
                print(f"PID {os.getpid()}: Computing interevent linear "
                      "transition matrices for {lamda=}, {t_s=}")

            t0 = time.time()
            for k, tk in enumerate(self.times[self._k_start_laplacians:
                                              self._k_stop_laplacians]):
                if verbose and not k%1000:
                    print(
                        f"PID {os.getpid()}: {k} over "
                        f"{self._k_stop_laplacians-1-self._k_start_laplacians}"
                    )
                    print(f"PID {os.getpid()} : {time.time()-t0:.2f}s")

                if self.instantaneous_events:
                    tau_k = 1.0
                else:
                    tau_k = self.times[self._k_start_laplacians+k+1] - tk

                Lcsr = self.laplacians[k].tocsr()

                if use_sparse_stoch:

                    #get non zero indices
                    nonzerosum_rowcols = ~np.logical_and(Lcsr.getnnz(1)==0,
                                                         Lcsr.getnnz(0)==0)

                    nz_rowcols, = (nonzerosum_rowcols).nonzero()

                    self.inter_T_lin[lamda][t_s].append(\
                           sparse_lin_approx(I - Lcsr,
                                             tau_k*lamda,
                                             Pi=self._stationary_trans[k],
                                             t_s=t_s,
                                             nz_rowcols=nz_rowcols))
                else:

                    self.inter_T_lin[lamda][t_s].append(\
                           lin_approx_trans_matrix(I - Lcsr,
                                                   tau_k*lamda,
                                                   Pi=self._stationary_trans[k],
                                                   t_s=t_s))

            if len(self.inter_T_lin[lamda][t_s]) == 0:
                if verbose:
                    print(f"PID {os.getpid()} no events, lin. trans. "
                          "matrix = identity")
                # is there was no event, the transition is identity
                if use_sparse_stoch:
                    self.inter_T_lin[lamda].append(
                        SparseStochMat.create_diag(size=self.num_nodes)
                    )
                else:
                    self.inter_T_lin[lamda][t_s].append(
                        eye(self.num_nodes, dtype=np.float64, format="csr")
                    )

            t_end = time.time()-t0
            self._compute_times[f"tran_matrices_lin_{lamda}_{t_s}"] = t_end

            if verbose:
                print(f"PID {os.getpid()}: Finished computing linear "
                      f"interevent transition matrices in {t_end=}")



    def compute_transition_matrices(self,
                                    lamda=None,
                                    t_start=None,
                                    t_stop=None,
                                    verbose=False,
                                    save_intermediate=True,
                                    reverse_time=False,
                                    force_csr=False,
                                    tol=None):
        """
        Compute transition matrices and saves them in a dict of lists
        `self.T[lamda]` where `self.T[lamda][k]` is the product of all
        interevent transition matrices from t_0 to t_k computed with lamda.
        """
        if not hasattr(self, "inter_T") or \
                lamda not in self.inter_T.keys():
            raise Exception("Compute inter_T first.")


        if not hasattr(self, "T"):
            self.T = dict()

        if reverse_time:
            k_init = len(self.inter_T[lamda])-1
            k_range = reversed(range(k_init))
            if verbose:
                print("PID ", os.getpid(), " : reversed time computation.")
        else:
            k_init = 0
            k_range = range(1,len(self.inter_T[lamda]))


        if lamda not in self.T.keys():
            if save_intermediate:
                if force_csr:
                    # forcing the first matrix to csr, will ensure that
                    # all products are done in csr format,
                    # since CSR @ SparseStochMat t is not implemented
                    self.T[lamda] = [self.inter_T[lamda][k_init].tocsr()]
                else:
                    self.T[lamda] = [self.inter_T[lamda][k_init]]

                if tol is not None:
                    set_to_zeroes(self.T[lamda][0],tol)
                    inplace_csr_row_normalize(self.T[lamda][0])
            else:
                if force_csr:
                    self.T[lamda] = self.inter_T[lamda][k_init].tocsr()
                else:
                    self.T[lamda] = self.inter_T[lamda][k_init]

                if tol is not None:
                    set_to_zeroes(self.T[lamda],tol)
                    inplace_csr_row_normalize(self.T[lamda])

            if verbose:
                print(f"PID {os.getpid()}: Computing transition matrix")

            t0 = time.time()

            for k in k_range:
                if verbose and not k%1000:
                    print(f"PID {os.getpid()}: {k} over "
                          f"{len(self.inter_T[lamda])}")
                    print(f"PID {os.getpid()} : {time.time()-t0:.2f}s")

                Tk = self.inter_T[lamda][k]
                if tol is not None:
                    set_to_zeroes(Tk,tol)
                    inplace_csr_row_normalize(Tk)

                if save_intermediate:

                    self.T[lamda].append(self.T[lamda][-1] @ Tk)

                    #normalize T to correct precision errors
                    if tol is not None:
                        set_to_zeroes(self.T[lamda][-1],tol)

                    inplace_csr_row_normalize(self.T[lamda][-1])
                else:
                    self.T[lamda] = self.T[lamda] @ Tk
                    if tol is not None:
                        set_to_zeroes(self.T[lamda], tol)

                    #normalize T to correct precision errors
                    inplace_csr_row_normalize(self.T[lamda])

            t_end = time.time()-t0

            self._compute_times[
                f"trans_matrix_{lamda}_rev{reverse_time}"
            ] = t_end

            if verbose:
                print("PID ", os.getpid(), " : ", f"finished in {t_end:.2f}s")
        elif verbose:
            print(f"PID {os.getpid()}: Transition matrices already computed "
                  f"for {lamda=}")

    def compute_lin_transition_matrices(self,
                                        lamda=None,
                                        t_start=None,
                                        t_stop=None,
                                        verbose=False,
                                        t_s=10,
                                        save_intermediate=True,
                                        reverse_time=False):
        """
        Compute transition matrices and saves them in a dict of lists
        `self.T_lin[lamda][t_s]` where `self.T_lin[lamda][t_s][k]` is the
        product of all interevent transition matrices from t_0 to t_k computed
        with lamda and t_s.
        """
        if not hasattr(self, "inter_T_lin") \
                or lamda not in self.inter_T_lin.keys() \
                or t_s not in self.inter_T_lin[lamda].keys():
            raise Exception("Compute inter_T_lin first.")

        if not hasattr(self, "T_lin"):
            self.T_lin = dict()

        compute = True

        if lamda in self.T_lin.keys():
            if t_s in self.T_lin[lamda].keys():
                compute = False
                if verbose:
                    print(f"PID {os.getpid()}: Transition matrices already "
                          f"computed for {lamda=}, {t_s=}")

        if compute:

            if reverse_time:
                k_init = len(self.inter_T_lin[lamda][t_s])-1
                k_range = reversed(range(k_init))
                if verbose:
                    print("PID ", os.getpid(), " : reversed time computation.")
            else:
                k_init = 0
                k_range = range(1,len(self.inter_T_lin[lamda][t_s]))


            # initial conditions
            if lamda not in self.T_lin.keys():
                self.T_lin[lamda] = dict()
                if save_intermediate:
                    self.T_lin[lamda][t_s] = [
                        self.inter_T_lin[lamda][t_s][k_init]
                    ]
                else:
                    self.T_lin[lamda][t_s] = \
                        self.inter_T_lin[lamda][t_s][k_init]

            if t_s not in self.T_lin[lamda].keys():
                if save_intermediate:
                    self.T_lin[lamda][t_s] = [
                        self.inter_T_lin[lamda][t_s][k_init]
                    ]
                else:
                    self.T_lin[lamda][t_s] = \
                        self.inter_T_lin[lamda][t_s][k_init]

            if verbose:
                print(f"PID {os.getpid()}: Computing transition matrix for "
                      f"{lamda=}, {t_s=}")

            t0 = time.time()

            for k in k_range:
                if verbose and not k%1000:
                    print(f"PID {os.getpid()}: {k} over "
                          f"{len(self.inter_T_lin[lamda][t_s])}")
                    print(f"PID {os.getpid()} : {time.time()-t0:.2f}s")
                if save_intermediate:
                    self.T_lin[lamda][t_s].append(self.T_lin[lamda][t_s][-1] @ \
                                      self.inter_T_lin[lamda][t_s][k])

                    #normalize T to correct precision errors
                    inplace_csr_row_normalize(self.T_lin[lamda][t_s][-1])
                else:
                    self.T_lin[lamda][t_s] = self.T_lin[lamda][t_s] @ \
                                      self.inter_T_lin[lamda][t_s][k]

                    #normalize T to correct precision errors
                    inplace_csr_row_normalize(self.T_lin[lamda][t_s])

            t_end = time.time()-t0

            self._compute_times[
                f"trans_matrix_lin_{lamda}_{t_s}_rev{reverse_time}"
            ] = t_end

            if verbose:
                print(f"PID {os.getpid()}: finished in {t_end=}")
        elif verbose:
            print(f"PID {os.getpid()}: Transition matrices already computed for"
                  f" {lamda=}")

    def _compute_stationary_transition(self,
                                       t_start=None,
                                       t_stop=None,
                                       verbose=False,
                                       use_sparse_stoch=True):

        if not hasattr(self, "laplacians"):
            self.compute_laplacian_matrices(t_start=t_start,
                                            t_stop=t_stop,
                                            verbose=verbose)
        if verbose:
            print(
                f"PID {os.getpid()}: Computing stationary transition matrices"
            )

        self._stationary_trans = []

        I = eye(self.num_nodes,
                dtype=np.float64, format="csc")

#        I = np.eye(self.num_nodes, dtype=np.float64)

        t0 = time.time()
        for k in range(len(self.laplacians)):
            if verbose and not k%1000:
                print(f"PID {os.getpid()}: {k} over {len(self.laplacians)}")
                print(f"PID {os.getpid()} : {time.time()-t0:.2f}s")

            if use_sparse_stoch:
                self._stationary_trans.append(\
                  sparse_stationary_trans(I - self.laplacians[k]))

            else:
                self._stationary_trans.append(\
                      compute_stationary_transition(I - self.laplacians[k]))


        t_end = time.time()-t0

        self._compute_times["_stationary_trans"] = t_end

        if verbose:
            print(f"PID {os.getpid()}: Stationary transition matrices "
                  f"computation took {t_end}s")


    def _merge_overlapping_events(self, verbose=False):
        """
        Merge temporally overlapping undirected event between each pair of
        nodes.
            
        """
        events_to_keep = np.ones(self.events_table.shape[0],dtype=bool)

        A = self.compute_static_adjacency_matrix()

        # loop over nodes
        for i,n1 in enumerate(self.nodes):

            for n2 in (A[n1,:] > 0).nonzero()[1]:

                mask_12 = np.logical_and(
                    self.events_table.source_nodes.values == n1,
                    self.events_table.target_nodes.values == n2
                )
                mask_21 = np.logical_and(
                    self.events_table.source_nodes.values == n2,
                    self.events_table.target_nodes.values == n1
                )

                #sort by starting times
                evs = self.events_table.loc[np.logical_or(
                    mask_12,
                    mask_21
                )].sort_values(by=[self._STARTS, self._ENDINGS])

                evs_list = list(evs.itertuples())

                # event to compare
                ev1 = evs_list[0]
                merged = 0
                for k in range(1,len(evs_list)):
                    ev2 = evs_list[k]
                    # if ev2 overlaps with ev1, merge them, otherwise ev2
                    # becomes ev1
                    if ev2.starting_times < ev1.ending_times:
                        #merge
                        events_to_keep[ev2.Index] = False
                        self.events_table.loc[
                            ev1.Index,
                            self._ENDINGS,
                        ] = ev2.ending_times
                        ev1._replace(ending_times=ev2.ending_times)
                        merged += 1
                    else:
                        ev1 = ev2
                if verbose:
                    print(f"n1,n2 ({n1},{n2}): {merged} merged")

        num_merged = (events_to_keep == False).sum()
        print("PID ", os.getpid(), " : ","merged ",
              num_merged, " events")

        self.events_table = self.events_table.loc[events_to_keep]

        self.events_table.reset_index(inplace=True, drop=True)

        self.num_events = self.events_table.shape[0]

        self.start_time = self.events_table[self._STARTS].min()

        self.end_time = self.events_table[self._ENDINGS].max()

        self._compute_time_grid()

        return num_merged


    def _compute_delta_trans_mat(self, lamda, round_zeros=True, tol=1e-8):
        """Comptes and put in a attribute `delta_inter_T` the matrix differences
        between each consecutive inter_T.
            
        self.delta_inter_T[lamda][k] = self.inter_T[lamda][k+1] - self.inter_T[lamda][k]
            
        The length of self.delta_inter_T[lamda] is len(self.inter_T[lamda]) - 1
            
        """
        if hasattr(self, "inter_T") and lamda in self.inter_T.keys():


            if not hasattr(self, "delta_inter_T"):
                self.delta_inter_T = dict()

            if lamda not in self.delta_inter_T.keys():

                self.delta_inter_T[lamda] = [
                    self.inter_T[lamda][k+1] - self.inter_T[lamda][k]
                    for k in range(len(self.inter_T[lamda])-1)
                ]

                if round_zeros:
                    for M in self.delta_inter_T[lamda]:
                        set_to_zeroes(M, tol=tol)

            else:
                print(f"PID {os.getpid()}: delta_inter_T has already been "
                      f"computed with {lamda=}")

        else:
            print(f"PID {os.getpid()}: delta_inter_T has not been computed")


        if hasattr(self, "inter_T_lin") and lamda in self.inter_T_lin.keys():

            if not hasattr(self, "delta_inter_T_lin"):
                self.delta_inter_T_lin = dict()

            if lamda not in self.delta_inter_T_lin.keys():

                self.delta_inter_T_lin[lamda] = dict()

                for t_s in self.inter_T_lin[lamda].keys():

                    self.delta_inter_T_lin[lamda][t_s] = [
                        self.inter_T_lin[lamda][t_s][k+1] - 
                            self.inter_T_lin[lamda][t_s][k]
                        for k in range(len(self.inter_T_lin[lamda][t_s])-1)
                    ]

                    if round_zeros:
                        for M in self.delta_inter_T_lin[lamda][t_s]:
                            set_to_zeroes(M, tol=tol)

            else:
                print(f"PID {os.getpid()}: delta_inter_T_lin has already been "
                      f"computed with {lamda=}")
        else:
            print(f"PID {os.getpid()}: delta_inter_T_lin has not been computed")

