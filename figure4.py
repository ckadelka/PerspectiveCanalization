#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 20:05:08 2025

@author: ckadelka
"""

import boolforge
import numpy as np
import matplotlib.pyplot as plt

n=4
 
all_functions = boolforge.get_left_side_of_truth_table(2**n)
 
strengths = []
input_redundancies = []
for function in all_functions:
    f = boolforge.BooleanFunction(function)
    strengths.append( f.get_canalizing_strength() )
    input_redundancies.append( f.get_input_redundancy() )
is_degenerate = []
for function in all_functions:
    f = boolforge.BooleanFunction(function)
    is_degenerate.append(f.is_degenerate())
 
strengths = np.array(strengths)
input_redundancies = np.array(input_redundancies)
is_degenerate = np.array(is_degenerate)
 

fig,ax = plt.subplots(figsize=(3,3))
ax.plot(strengths[~is_degenerate],input_redundancies[~is_degenerate],'bx')
#ax.plot(strengths[is_degenerate],input_redundancies[is_degenerate],'bx')
ax.set_xlabel('canalizing strength')
ax.set_ylabel('normalized input redundancy')
ax.spines[['right', 'top']].set_visible(False)
plt.savefig(f'strength_vs_redundancy_n{n}.pdf',bbox_inches = "tight")