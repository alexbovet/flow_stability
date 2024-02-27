"""
#
# flow stability
#
# Copyright (C) 2021 Alexandre Bovet <alexandre.bovet@maths.ox.ac.uk>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


"""

import numpy as np
from queue import Empty, PriorityQueue
from scipy.stats import expon
from scipy.sparse import lil_matrix


#%%


class Distro(object):
    """ Random variables from distributions 
    
        The loc and scale values are the default values used, but
        they can also be changed when calling `Distro.draw_val(loc,scale)`
    """
    def __init__(self, loc=0, scale=1, dist_type='exponential'):
        self.loc=loc
        self.scale=scale
        self.dist_type=dist_type
        
        if dist_type == 'exponential':
            self.draw = expon.rvs
        else:
            raise NotImplementedError('Not implemented distribution')
            
    def draw_val(self, loc=None, scale=None):
        if loc is None:
            loc=self.loc
        if scale is None:
            scale=self.scale
            
        return self.draw(loc=loc, scale=scale)
            

class Individual(object):
    """ Individual agent class.
    
        Parameters:
        -----------
        
        ID: int
            ID of the individual
            
        inter_distro_loc: float
            Location (mean) of the interactions (i.e. events) durations
            
        inter_distro_scale: float
            Scale (standard-deviation) of the interactions (i.e. events) durations
            
        inter_distro_type: string
            Type of distribution function of the interactions durations.
            Can be "exponential".
            
        inter_distro_mod_func: function
            function that take an argument `time` and returns a tuple 
            `(loc, scale)` used as parameters for drawing time-dependent 
            interaction durations.
            
        activ_distro_loc: float
            Location (mean) of the inter-activation (i.e. inter-events) durations
            
        activ_distro_scale: float
            Scale (standard-deviation) of the inter-activation (i.e. inter-events) durations
            
        activ_distro_type: string
            Type of distribution function of the inter-activation durations.
            Can be "exponential".
            
        activ_distro_mod_func: function
            function that take an argument `time` and returns a tuple 
            `(loc, scale)` used as parameters for drawing time-dependent 
            inter-activation durations.
            
        group: int
            ID of the group to which the individual belongs to.
    
    """
    # these are class attributes used to keep track of parameters of all instances
    all_IDs = []
    all_groups = []
    def __init__(self, ID,
                 inter_distro_loc=0,
                 inter_distro_scale=1, 
                 inter_distro_type="exponential",
                 inter_distro_mod_func=None,
                 activ_distro_loc=0,
                 activ_distro_scale=5,
                 activ_distro_type="exponential",
                 activ_distro_mod_func=None,
                 group=0):
        # ID of the individual (must be unique)        
        # group 
        if not isinstance(ID,int):
            raise ValueError('ID must be an integer.')
        
        self.ID = ID
        Individual.all_IDs.append(ID)
        
        # distribution of interaction durations
        self.inter_distro = Distro(scale=inter_distro_scale,
                             loc=inter_distro_loc,
                             dist_type=inter_distro_type)
        
        # distribution of inter-activation times
        self.activ_distro = Distro(scale=activ_distro_scale,
                             loc=activ_distro_loc,
                             dist_type=activ_distro_type)
        
        # loc, scale modulation functions
        if inter_distro_mod_func is None:
            self.inter_distro_mod_func = lambda _ : (inter_distro_loc,
                                                inter_distro_scale)
        else:
            self.inter_distro_mod_func = inter_distro_mod_func
        
        if activ_distro_mod_func is None:
            self.activ_distro_mod_func = lambda _ : (activ_distro_loc,
                                                activ_distro_scale)
        else:
            self.activ_distro_mod_func = activ_distro_mod_func
        

        # group 
        if not isinstance(group,int):
            raise ValueError('group must be an integer.')
        
        self.group = group
        
        Individual.all_groups.append(group)
        
        # initialize simulation time
        self.t = 0

    def draw_inter_duration(self, time=None):
        """ draws a interaction duration time from `inter_distro`.
        
            If `time` is provided, computes the `loc` and `scale` 
            parameters using `inter_distro_mod_func(time)`.
            Otherwise, `loc` and `scale` are taken as the 
            initialized values.

        """
        
        if time is None:
            return self.inter_distro.draw_val()

        else:
            loc, scale = self.inter_distro_mod_func(time)
            if scale == 0:
                raise ValueError('distribution scale cannot be zero.')
                
            return self.inter_distro.draw_val(loc=loc, scale=scale)
        
            
    def draw_activ_time(self, time=None):
        """ draws a activation time from `activ_distro`.
        
            If `time` is provided, computes the `loc` and `scale` 
            parameters using `activ_distro_mod_func(time)`.
            Otherwise, `loc` and `scale` are taken as the 
            initialized values.

        """
        
        if time is None:
            return self.activ_distro.draw_val()

        else:
            loc, scale = self.activ_distro_mod_func(time)
            if scale == 0:
                raise ValueError('distribution scale cannot be zero.')
                
            return self.activ_distro.draw_val(loc=loc, scale=scale)
        
        
class SynthTempNetwork(object):
    """ SynthTempNetwork: a class for an agent based model 
        generating a continuous time synthetic temporal network
        
        Alexandre Bovet 2019
        
        Parameters:
        -----------
         
        individuals: list
            List of Individual instances, i.e. the nodes of the network.
            Individual have a group id. There are N individuals and Ngroups group.
         
        t_start: float
            Starting time of the simulation
            
        t_end: float
            Ending time of the simulation
            
        num_interactions_per_activation: int
            Number of interactions generated each time an individual is
            activated
            
        next_event_method: string
            Method to choose the individual to interact with.
            Can be
            
            - 'random_uniform' (default): 
                uniform probability to choose any other individuals.
            - 'block_probs':
                probabilities given by a block matrix  `inter_group_probs`.
            - 'block_probs_mod': 
                probabilities given by a time-dependent function `block_prob_mod_func`.
                
        inter_group_probs: Ngroups x Ngroups numpy array
            Contains the probabilities of inter-group interactions.
            
        block_prob_mod_func: function
            Functions that depend on t such that `block_prob_mod_func(t)` returns
            an `inter_group_probs` matrix.
            
        Usage:
        ------
        
        The simulation is run by calling `self.run()`. All the events are stored
        in 4 lists: `self.indiv_sources`, `self.indiv_targets`, 
        `self.start_times` and `self.end_times`.
        
    """
    def __init__(self, individuals, t_start=0, t_end=200, 
                 num_interactions_per_activation=1,
                 next_event_method='random_uniform',
                 inter_group_probs=None,
                 block_prob_mod_func=None):

        # number of individuals
        self.N = len(individuals)
        
        # list of individuals
        self.individuals = individuals
        
        # list of individual IDs
        self.indiv_ids_list = [ind.ID for ind in individuals]

        if not all([i < self.N for i in self.indiv_ids_list]):
            raise ValueError('individual IDs must be from 0 to N-1')
        
        # array of individual IDs
        self.indiv_ids_array = np.array(self.indiv_ids_list, dtype=np.int32)
        
        try:
            assert np.unique(self.indiv_ids_array).size == self.N
        except Exception as e:
            print('Individuals ID must be unique.')
            raise e
        
        # individuals group list
        self.indiv_groups_list = [ind.group for ind in individuals]

        # individuals group array
        self.indiv_groups_array = np.array(self.indiv_groups_list, dtype=np.int32)

        # number of groups
        self.Ngroups = len(set(self.indiv_groups_list))

        if not all([gid < self.Ngroups for gid in self.indiv_groups_list]):
            raise ValueError('group IDs must be from 0 to Ngroups-1')

        # group id to indiv ids dict
        self.group_to_ids = {g : np.where(self.indiv_groups_array == g)[0] \
                                for g in np.unique(self.indiv_groups_array)}
        
        # final and start time of the simulation
        self.t_end = t_end
        self.t_start = t_start
        
        # number of interactions per activations
        self.num_interactions_per_activation = num_interactions_per_activation
        
        # Priority queue for the discrete event simulation
        self.queue = PriorityQueue()
        
        # dictionary mapping indiv_id to their event in the Priority Queue
        # Each individual has at most one event in the queue 
        self.event_mapper_activ = {}
        self.event_mapper_inter = {}

        # how events are chosen
        self.next_event_method = next_event_method

        # inter group interaction probabilities matrix
        # p[i,j] = prob of group i interacting with j
        self.inter_group_probs = inter_group_probs
        if isinstance(inter_group_probs, list):
            self.inter_group_probs = np.array(inter_group_probs)

        if next_event_method == 'block_probs' and \
                        self.inter_group_probs.shape != (self.Ngroups, self.Ngroups):
            raise ValueError('inter_group_probs must have (Ngroups,Ngroups) shape' + \
                              'for "block_probs" next_event_method')
        
        if next_event_method == 'block_probs':
            # make sure inter_group_probs is properly normalized
            # i.e. sum_j p[i,j] = 1 for all i
            self.inter_group_probs = \
                (self.inter_group_probs.T/self.inter_group_probs.sum(axis=1)).T
                
        if next_event_method == 'block_probs_mod':
            if block_prob_mod_func is None:
                raise ValueError('`block_prob_mod_func` must be specified for the ' + \
                                 'method `block_prob_mod_func`.')
            else:
                self.block_prob_mod_func = block_prob_mod_func

        # events information : start time, length and box number
        # a list for each individual -> list of lists
        self.indiv_sources = []
        self.indiv_targets = []
        self.start_times = []
        self.end_times = []

        # instantaneous adjacency matrix
        self._A = lil_matrix((self.N, self.N),
               dtype=np.int32)

        # sparse matrix to store last interaction starting events
        self._last_times = lil_matrix((self.N, self.N),
               dtype=np.float64)

    @staticmethod
    def Event(time, indiv_id, event_type='activation', partner=None, is_canceled=False):
        """
        Arguments:
            - time: float, used to order the events in the priority queue
            - indiv_id: int, id of the individual
            - event_type: str, type of event, 'activation' (default) or 'interaction'
            - is_canceled : bool, =True if the event has been replaced and must be discarded
                            This parameter is changed through the event_mapper dict.
        """
        if event_type == 'activation':
            return time, [indiv_id, event_type, is_canceled]
        if event_type == 'interaction':
            return time, [indiv_id, event_type, partner, is_canceled]
        else:
            raise NotImplementedError('Unknown event_type {0}.'.format(event_type))
    
        
    def put_event(self, indiv_id,
                  event_type='activation',
                  partner=None,
                  is_instantaneous=False):
        """ put an event in the priority queue accorind to the rules
            event_type can be 'activation' or 'interaction'
        """

        if event_type == 'activation':
            
            next_activation_time = self.individuals[indiv_id].draw_activ_time(self.t)

            event = self.Event(time=self.t + next_activation_time,
                                indiv_id=indiv_id,
                                event_type=event_type) 

            # map event to indiv_id
            self.event_mapper_activ[indiv_id] = event 

        if event_type == 'interaction':
            
            if is_instantaneous:
                event_length = 0
            else:
                event_length = self.individuals[indiv_id].draw_inter_duration(self.t)
            
            event = self.Event(time=self.t + event_length,
                                indiv_id=indiv_id,
                                event_type=event_type,
                                partner=partner) 
            self.event_mapper_inter[indiv_id] = event

        self.queue.put_nowait(event)
        
    def get_new_partner(self, indiv_id, current_partner=None):
        """ compute the next partner"""
        
        if self.next_event_method == 'random_uniform':

            # unallowed new partners (self+existing)
            un_partners = {indiv_id}
            for n in self._A[indiv_id,:].nonzero()[1]:
                un_partners.add(n)
                
                
            potential_partners = self.indiv_ids_array
            if set(potential_partners) == un_partners:
                new_partner = None
            else:                                   
                new_partner = indiv_id
                while new_partner in un_partners:
                    new_partner = \
                        np.random.choice(potential_partners)   

        elif self.next_event_method == 'random_uniform_within_group':

            # unallowed new partners (self+existing)
            un_partners = {indiv_id}
            for n in self._A[indiv_id,:].nonzero()[1]:
                un_partners.add(n)
                
            potential_partners = self.group_to_ids[self.indiv_groups_array[indiv_id]]
            if set(potential_partners) == un_partners:
                new_partner = None
            else:                   
                new_partner = indiv_id
                while new_partner in un_partners:
                    new_partner = \
                        np.random.choice(potential_partners)                    

        elif self.next_event_method == 'block_probs':
            # block model type interactions
            indiv_group_id = self.indiv_groups_array[indiv_id]

            # draw target interacting group
            
            #prob of picking groups
            p = self.inter_group_probs[indiv_group_id,:]
            if p.sum() == 0:
                # no possible partners
                new_partner = None
            else:
                target_group = np.random.choice(np.arange(self.Ngroups),
                                    p=p)
    
                # unallowed new partners (self+existing)
                un_partners = {indiv_id}
                for n in self._A[indiv_id,:].nonzero()[1]:
                    un_partners.add(n)
                    
                potential_partners = self.group_to_ids[target_group]
                if (set(potential_partners).union(set([indiv_id]))).issubset(un_partners):
                    new_partner = None
                else:                    
                    new_partner = indiv_id
                    while new_partner in un_partners:
                        new_partner = \
                            np.random.choice(potential_partners)

        elif self.next_event_method == 'block_probs_mod':
            # block model type interactions with time dependence
            indiv_group_id = self.indiv_groups_array[indiv_id]

            inter_group_probs = self.block_prob_mod_func(self.t)
            
            # draw target interacting group
            #prob of picking groups
            p = inter_group_probs[indiv_group_id,:]
            if p.sum() == 0:
                # no possible partners
                new_partner = None
            else:                
                target_group = np.random.choice(np.arange(self.Ngroups),
                                    p=p)
    
                # unallowed new partners (self+existing)
                un_partners = {indiv_id}
                for n in self._A[indiv_id,:].nonzero()[1]:
                    un_partners.add(n)
                    
                
                potential_partners = self.group_to_ids[target_group]
                if (set(potential_partners).union(set([indiv_id]))).issubset(un_partners):
                    new_partner = None
                else:
                    new_partner = indiv_id
                    while new_partner in un_partners:
                        new_partner = \
                            np.random.choice(potential_partners)
        else:
            raise NotImplementedError('next_event_method :' +\
                '{} is not implemented yet.'.format(self.next_event_method))
        return new_partner
        
    
    def initialize_events(self):
        """ put one activation event in the queue for each individual"""
        for indiv_id in self.indiv_ids_list:
            
            self.put_event(indiv_id, event_type='activation')

                
    def run(self, save_all_states=False, save_dt_states=False, dt=10.0, verbose=False):
        """ run the simulation
        
        Parameters
        ----------
        save_all_states : bool
            saves the positions in `all_states` for each new event        
        save_dt_states : bool
            saves the positions in `saved_states` at a constant frequency given by `dt`
        dt : float
            record period for `saved_states' 
        verbose : bool
            verbose mode shows the progress
        """
    
        # time interval at which the data is recorded
        self.dt= dt
        
        # simulation time
        self.t=self.t_start

        self._t_next_bin = self.t + self.dt
        
        # initialize the simulation
        self.initialize_events()       
        
        # save initial conditions
        if save_all_states:
            # save positions at every event time
            self.all_states = dict()
            self.all_states[self.t] = self._A.copy()    
        if save_dt_states:
            # save positions every dt
            self.saved_states = dict()
            self.saved_states[self.t] = self._A.copy()

        # advance the simulation                
        while self.t < self.t_end:
            try:
                prev_time = self.t
                
                (time, event) = self.queue.get_nowait()
                
                if verbose:
                    print("treating event : ")
                    print(((time, event)))
                    print('--')

                    
                if not event[-1]: # not is_canceled
                    self.t = time
                    
                    if event[1] == 'activation':
                        [indiv_id, _, _] = event
                        
                        # put next event in the queue 
                        self.put_event(indiv_id, event_type='activation')

                        for _ in range(self.num_interactions_per_activation):
                            # pick new partner for interaction
                            partner = self.get_new_partner(indiv_id)
    
                            if verbose>10:
                                print("new interaction between : ", indiv_id, ' and ', partner)
                                print('starting at ', self.t)
                                print('--')
                    
                            if partner is not None:
                                #update A and _last_times
                                self._A[indiv_id,partner] = 1
        
                                self._last_times[indiv_id,partner] = self.t
        
                                self.put_event(indiv_id, event_type='interaction',
                                                partner=partner)
                            else:
                                pass
                                # if partner is None, continue activations but
                                # there is no interaction
                            
                    elif event[1] == 'interaction':
                        # end of an interaction
                        [indiv_id, _, partner, _] = event

                        #update A
                        self._A[indiv_id,partner] = 0

                        # starting time of this event
                        event_start = self._last_times[indiv_id,partner]

                        # erase last_time
                        self._last_times[indiv_id,partner] = 0
                        
                        # add new event to lists
                        self.indiv_sources.append(indiv_id)
                        self.indiv_targets.append(partner)
                        self.start_times.append(event_start)
                        self.end_times.append(self.t)
    
                    # if the simulation time has advanced
                    if self.t > prev_time:
                        
                        if save_all_states:
                            # save positions at every new times
                            self.all_states[self.t] = self._A.copy()
                        
                        if save_dt_states:
                            # save positions every dt 
                            if self.t >= self._t_next_bin:
                                # save positions
                                self.saved_states[self.t] = self._A.copy()
                                self._t_next_bin += self.dt

                                if verbose:
                                    print("t = ", self.t)

                
            except Empty:
                print('Priority queue is empty')
                break
            

#%%

if __name__ == '__main__':
    # Simple example
    individuals = [Individual(i, group=0) for i in range(20)]
    sim = SynthTempNetwork(individuals, t_start=0, t_end=50) 

    sim.run(save_all_states=True, save_dt_states=True, verbose=True)

    # more advanced example
    import time
    inter_tau = 1
    activ_tau = 2
    t_start = 0
    n_groups = 3
    n_per_group = 9
    individuals = []
    
    
    def make_step_block_probs(deltat1, deltat2, m1=1, p1=1):
        """ `deltat1` is the length of the echanging step
        
            `deltat2` is the length of the within step
            
            `m1` is the prob of self-interaction
            
            `p1` is the prob of cross-interaction
        """
    
    
        def block_mod_func(t):
            
            m2 = (1-m1)/2
            p2 = (1-p1)
            
            ex12 = np.array([[p2,p1,0],
                             [p1,p2,0],
                             [0,0,0]])
            ex23 = np.array([[0,0,0],
                             [0,p2,p1],
                             [0,p1,p2]])
            ex13 = np.array([[p2,0,p1],
                             [0, 0, 0],
                             [p1,0,p2]])
        
            I = np.array([[m1,m2,m2],
                          [m2,m1,m2],
                          [m2,m2,m1]])
            if t>=0 and  t < deltat1:
                return I
            elif t>=deltat1 and t<deltat1+deltat2:
                return ex12
            elif t>=deltat1+deltat2 and t < 2*deltat1+deltat2:
                return I
            elif t>= 2*deltat1+deltat2 and t < 2*(deltat1+deltat2):
                return ex23
            elif t>= 2*(deltat1+deltat2) and t < 2*(deltat1+deltat2)+deltat1:
                return I
            elif t>=2*(deltat1+deltat2)+deltat1 and t <= 3*(deltat1+deltat2):
                return ex13
            else:
                print('Warning : t must be >=0 and <= 3*(deltat1+deltat2)' +\
                      't is ', t)
                return I
            
        return block_mod_func
     
    m1=0.8
    p1=0.8
    deltat1 = 40*activ_tau
    deltat2=(9/2*m1-3/2)*deltat1/(2*p1-1)
    
    t_end = 3*(deltat1+deltat2)
    
    block_prob_mod_func = make_step_block_probs(deltat1,deltat2,m1,p1)
    
    for g in range(n_groups):
    
        individuals.extend([Individual(i, inter_distro_scale=inter_tau,
                                          activ_distro_scale=activ_tau,
                                          group=g) for i in range(g*n_per_group,(g+1)*n_per_group)])
    
    
    sim = SynthTempNetwork(individuals=individuals, t_start=t_start, t_end=t_end,
                           next_event_method='block_probs_mod',
                           block_prob_mod_func=block_prob_mod_func)
    
    print('running simulation')
    t0 = time.time()
    sim.run(save_all_states=True, save_dt_states=True)
    print('done in ', time.time()-t0)
#%%
