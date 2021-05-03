#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:02:45 2021

@author: Alexandre Bovet
"""

from SynthTempNetwork import Individual, SynthTempNetwork
from TemporalNetwork import ContTempNetwork


#%% Paper example

inter_tau = 1
activ_tau = 1
t_start = 0
n_groups = 3
n_per_group = 9
individuals = []


m1=0.95
p1=0.95
deltat1 = 40
deltat2=120

t_end = 3*(deltat1+deltat2)

def make_step_block_probs(deltat1, deltat2, m1=1, p1=1):
    """ `deltat1` is the length of the echanging step
    
        `deltat2` is the length of the within step
        
        `m1` is the prob of self-interaction (during deltat1)
        
        `p1` is the prob of cross-interaction (during deltat2)
    """


    def block_mod_func(t):
        
        m2 = (1-m1)/2
        p2 = (1-p1)
        
        ex12 = np.array([[p2,p1,0],
                         [p1,p2,0],
                         [0,0,1]])
        ex23 = np.array([[1,0,0],
                         [0,p2,p1],
                         [0,p1,p2]])
        ex13 = np.array([[p2,0,p1],
                         [0, 1, 0],
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
 


block_prob_mod_func = make_step_block_probs(deltat1,deltat2,m1,p1)

for g in range(n_groups):

    individuals.extend([Individual(i, inter_distro_scale=inter_tau,
                                      activ_distro_scale=activ_tau,
                                      group=g) for i in range(g*n_per_group,(g+1)*n_per_group)])


    
#%%
with open(os.path.join(savedir, filename + '_sim_param.pickle'), 'wb') as fopen:
    pickle.dump({'inter_tau' : inter_tau,
                'activ_tau' : activ_tau,
                't_start' : t_start,
                't_end' : t_end,
                'n_groups' : n_groups,
                'n_per_group' : n_per_group,
                'deltat1': deltat1,
                'deltat2' : deltat2,
                'p1' : p1,
                'm1' : m1,
                }, fopen)
#%% run sim    
sim = SynthTempNetwork(individuals=individuals, t_start=t_start, t_end=t_end,
                       next_event_method='block_probs_mod',
                       block_prob_mod_func=block_prob_mod_func)

print('running simulation')
t0 = time.time()
sim.run(save_all_states=True, save_dt_states=True, verbose=False)
print('done in ', time.time()-t0)
#%%
net = ContTempNetwork(source_nodes=sim.indiv_sources,
                      target_nodes=sim.indiv_targets,
                      starting_times=sim.start_times,
                      ending_times=sim.end_times,
                      merge_overlapping_events=True)