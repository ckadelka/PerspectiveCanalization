#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 09:56:16 2025

@author: ckadelka
"""
import boolforge
import numpy as np
import matplotlib.pyplot as plt


def fak(n):
    if n==0 or n==1:
        return 1
    else:
        return n*fak(n-1)

def ndbinom(n,ks):
    if len(ks)==1:
        return nchoosek(n,ks[0])
    res= 1
    for k in ks:
        res*=fak(k)
    return fak(n)/res

def nbinom(n,k):
    return fak(n)/(fak(k)*fak(n-k))

def Bstar(n):
    return 2**(2**n) - 2*((-1)**n -n) + sum([(-1)**k*nbinom(n,k)*2**(k+1)*2**(2**(n-k)) for k in range(1,n+1)])


def nchoosek(population, sample):
    "Returns `population` choose `sample`."
    s = max(sample, population - sample)
    assert s <= population
    assert population > -1
    if s == population:
        return 1
    numerator = 1
    denominator = 1
    for i in range(s+1, population + 1):
        numerator *= i
        denominator *= (i - s)
    return numerator/denominator

def sterling_times_fak_r(n,r):
    return sum([((-1)**i)*nchoosek(r,i)*(r-i)**n for i in range(r+1)])
    
def sterling_difference(n,r):
    return r**n+sum([(-1)**i*nchoosek(r-1,i-1)*(r-i)**(n-1)*(r**2*1./i-r+n) for i in range(1,r+1)])

def number_ncfs(n,p):
    if n==1 and p==2:
        return (2,0,2)
    if p==2:
        N1=0
    else:
        N1=2**(n-1)*p*(p-2)*(p-1)**n*sum([(p-1)**(r-1)*n*sterling_times_fak_r(n-1,r-1) for r in range(2,n+1)])
    #N2=2**n*p*(p-1)**n*sum([(p-1)**r*(sterling_times_fak_r(n,r)-n*sterling_times_fak_r(n-1,r-1)) for r in range(1,n)])
    N2=2**n*p*(p-1)**n*sum([(p-1)**r*sterling_difference(n,r) for r in range(1,n)])
    return N1+N2,N1,N2

def partitions(n, I=1):
    yield (n,)
    for i in range(I, n//2 + 1):
        for p in partitions(n-i, i):
            yield (i,) + p

def number_k_canalizing_depth(n,k):
    return nchoosek(n,k)*(number_ncfs(k,2)[0]+Bstar(n-k)*2**(k+1)*sum([ndbinom(k,ks) for ks in partitions(k)])    )




n_min,n_max = 0,5
ns = np.arange(n_min,n_max+1)
canalizing_depths = np.arange(max(ns)+1)
proportion_canalizing_depths = np.zeros((len(ns),max(ns)+1))

for i,n in enumerate(ns):
    number_f_with_canalizing_depths = [number_k_canalizing_depth(n,k) for k in range(1,n+1)]
    total_fs = 2**(2**n)
    number_f_not_canalizing = total_fs-sum(number_f_with_canalizing_depths)
    proportion_canalizing_depths[i,:n+1] = np.append(number_f_not_canalizing,number_f_with_canalizing_depths)/total_fs

fig,ax = plt.subplots()
for i,canalizing_depth in enumerate(canalizing_depths):
    ax.bar(ns,proportion_canalizing_depths[:,i],bottom=np.sum(proportion_canalizing_depths[:,:i],1),label=str(canalizing_depth))
ax.legend(frameon=False,loc='center',bbox_to_anchor=[0.5,1.1],ncol=8,title='exact canalizing depth of Boolean function')
ax.set_xticks(ns)
ax.set_xlabel('Number of inputs')
ax.set_ylabel('Proportion of functions')
ax.set_ylim([0,1])
plt.savefig(f'depth_vs_variables_n{n_min}to{n_max}.pdf',bbox_inches='tight')






n_max = 5 #n_min must be 0
ns = np.arange(n_max+1)
canalizing_depths = np.arange(max(ns)+1)
proportion_canalizing_depths = np.zeros((len(ns),max(ns)+1))
C_n_k = np.zeros((len(ns),max(ns)+1),dtype=int)
N_n_m_k = np.zeros((len(ns),max(ns)+1,max(ns)+1),dtype=int)

for n in range(n_max+1):
    C_n_k[n,1:n+1] = [number_k_canalizing_depth(n,k) for k in range(1,n+1)]
    total_fs = 2**(2**n)
    C_n_k[n,0] = total_fs-sum(C_n_k[n,1:n+1])
    for m in range(n+1): #number essential variables
        for k in range(m+1): #depth
            if m == k and k == 0:
                N_n_m_k[n,m,k] = 2
            elif k <= m and m < n:
                N_n_m_k[n,m,k] = nchoosek(n,m) * N_n_m_k[m,m,k]
            elif m == n:
                N_n_m_k[n,m,k] = C_n_k[n,k] - sum([N_n_m_k[n,i,k] for i in range(n)])

    proportion_canalizing_depths[n] = N_n_m_k[n,n]/sum(N_n_m_k[n,n])
                
fig,ax = plt.subplots()
for i,canalizing_depth in enumerate(canalizing_depths):
    ax.bar(ns,proportion_canalizing_depths[:,i],bottom=np.sum(proportion_canalizing_depths[:,:i],1),label=str(canalizing_depth))
ax.legend(frameon=False,loc='center',bbox_to_anchor=[0.5,1.1],ncol=8,title='exact canalizing depth of non-degenerate Boolean function')
ax.set_xticks(ns)
ax.set_xlabel('Number of essential inputs')
ax.set_ylabel('Proportion of functions')
ax.set_ylim([0,1])
plt.savefig(f'depth_vs_essential_variables_n{n_min}to{n_max}.pdf',bbox_inches='tight')
















nsim = 1000
EXACT_DEPTH = True #Default: False
ns = np.arange(2,6)
canalizing_strengths = np.zeros((len(ns),max(ns)+1,nsim))
input_redundancies = np.zeros((len(ns),max(ns)+1,nsim))

for k in range(nsim):
    for i,n in enumerate(ns):
        for depth in np.append(np.arange(n-1),n):
            f = boolforge.random_function(n,depth=depth,EXACT_DEPTH=EXACT_DEPTH)
            canalizing_strengths[i,depth,k] = f.get_canalizing_strength()
            input_redundancies[i,depth,k] = f.get_input_redundancy()

# width = 0.28
# violinplot_args = {'widths': width, 'showmeans': True, 'showextrema': False}
# fig, ax = plt.subplots(2, 1, figsize=(5,5), sharex=True)

# base_gap = 1     # gap between groups
# intra_gap = 0.3   # gap within group

# max_depth = max(ns)

# for ii, (data, label) in enumerate(zip(
#     [canalizing_strengths, input_redundancies],
#     ['canalizing strength', 'normalized input redundancy'])):

#     #ax[ii].grid(axis='y')
#     ax[ii].spines[['right', 'top']].set_visible(False)

#     positions = []
#     values = []
#     colors_used = []
#     group_centers = []

#     current_x = 0.0
#     for i, n in enumerate(ns):
#         valid_depths = np.append(np.arange(n-1), n)
#         n_viols = len(valid_depths)

#         # positions centered on each group's midpoint
#         offsets = np.linspace(
#             -(n_viols - 1) * intra_gap / 2,
#             (n_viols - 1) * intra_gap / 2,
#             n_viols
#         )
#         group_positions = current_x + offsets
#         positions.extend(group_positions)
#         group_centers.append(current_x)

#         for depth in valid_depths:
#             values.append(data[i, depth, :])
#             colors_used.append('C'+str(depth))

#         # advance x-position based on total group width
#         group_width = (n_viols - 1) * intra_gap
#         current_x += group_width / 2 + base_gap + width + intra_gap

#     # plot violins one by one with colors
#     for vpos, val, c in zip(positions, values, colors_used):
#         vp = ax[ii].violinplot(val, positions=[vpos], **violinplot_args)
#         for body in vp['bodies']:
#             body.set_facecolor(c)
#             body.set_alpha(0.85)
#         vp['cmeans'].set_color('k')

#     # axis labels
#     ax[ii].set_ylabel(label)
#     if ii == 1:
#         ax[ii].set_xlabel('Number of non-degenerate inputs (n)')
#         ax[ii].set_xticks(group_centers)
#         ax[ii].set_xticklabels(ns)
#     ax[ii].set_ylim([-0.02,1.02])

# # add legend for depth colors
# depth_handles = [
#     plt.Line2D([0], [0], color='C'+str(d), lw=5, label=f'{d}')
#     for d in range(max_depth + 1)
# ]
# a=fig.legend(handles=depth_handles, loc='upper center', ncol=7, frameon=False,
#              title='exact canalizing depth of Boolean function' if EXACT_DEPTH else 'minimal canalizing depth')
# plt.savefig('all_can.pdf',bbox_inches='tight')





observed_proportion_canalizing_depths = [[2,0,0,0,0,0],
                                         [0,2097,0,0,0,0],
                                         [0,0,1288,0,0,0],
                                         [24,3,0,660,0,0],
                                         [45,5,0,0,400,0],
                                         [33,17,3,0,0,182]]
observed_proportion_canalizing_depths = np.array(observed_proportion_canalizing_depths)
observed_proportion_canalizing_depths = observed_proportion_canalizing_depths/observed_proportion_canalizing_depths.sum(axis=1)[:,None]

#all together

width = 0.28
violinplot_args = {'widths': width, 'showmeans': True, 'showextrema': False}
fig, ax = plt.subplots(5, 1, figsize=(5,9.5), height_ratios=[2,2,2,3,3], sharex=True)

base_gap = 1     # gap between groups
intra_gap = 0.3   # gap within group

max_depth = max(ns)



ax[0].spines[['right', 'top']].set_visible(False)
ax[1].spines[['right', 'top']].set_visible(False)
ax[2].spines[['right', 'top']].set_visible(False)

positions = []
values_expected = []
values_observed = []
colors_used = []
group_centers = []

current_x = 0.0
for i, n in enumerate(ns):
    valid_depths = np.append(np.arange(n-1), n)
    n_viols = len(valid_depths)

    # positions centered on each group's midpoint
    offsets = np.linspace(
        -(n_viols - 1) * intra_gap / 2,
        (n_viols - 1) * intra_gap / 2,
        n_viols
    )
    group_positions = current_x + offsets
    positions.extend(group_positions)
    group_centers.append(current_x)

    for depth in valid_depths:
        values_expected.append(proportion_canalizing_depths[n, depth])
        values_observed.append(observed_proportion_canalizing_depths[n,depth])
        colors_used.append('C'+str(depth))

    # advance x-position based on total group width
    group_width = (n_viols - 1) * intra_gap
    current_x += group_width / 2 + base_gap + width + intra_gap

    # axis labels
    ax[1].set_xticks(group_centers)
    ax[2].set_xticks(group_centers)

# plot violins one by one with colors
for vpos, val, c in zip(positions, values_observed, colors_used):
    bar = ax[0].bar([vpos], val, color=c,width=width)
for vpos, val, c in zip(positions, values_expected, colors_used):
    bar = ax[1].bar([vpos], val, color=c,width=width)
    bar = ax[2].bar([vpos], val, color=c,width=width)
ax[2].set_yscale('log')
ax[1].set_ylabel('expected\nproportion')
ax[2].set_ylabel('expected\nproportion [log]')
ax[2].set_yticks([1.e-06, 1.e-04, 1.e-02, 1])
ax[2].set_yticklabels(['$\\mathdefault{10^{-6}}$','$\\mathdefault{10^{-4}}$','$\\mathdefault{10^{-2}}$','1'])
ax[0].set_ylabel('observed\nproportion')



for ii, (data, label) in enumerate(zip(
    [canalizing_strengths, input_redundancies],
    ['canalizing strength', 'normalized\ninput redundancy'])):

    #ax[ii].grid(axis='y')
    ax[ii+3].spines[['right', 'top']].set_visible(False)

    positions = []
    values = []
    colors_used = []
    group_centers = []

    current_x = 0.0
    for i, n in enumerate(ns):
        valid_depths = np.append(np.arange(n-1), n)
        n_viols = len(valid_depths)

        # positions centered on each group's midpoint
        offsets = np.linspace(
            -(n_viols - 1) * intra_gap / 2,
            (n_viols - 1) * intra_gap / 2,
            n_viols
        )
        group_positions = current_x + offsets
        positions.extend(group_positions)
        group_centers.append(current_x)

        for depth in valid_depths:
            values.append(data[i, depth, :])
            colors_used.append('C'+str(depth))

        # advance x-position based on total group width
        group_width = (n_viols - 1) * intra_gap
        current_x += group_width / 2 + base_gap + width + intra_gap

    # plot violins one by one with colors
    for vpos, val, c in zip(positions, values, colors_used):
        vp = ax[ii+3].violinplot(val, positions=[vpos], **violinplot_args)
        for body in vp['bodies']:
            body.set_facecolor(c)
            body.set_alpha(0.85)
        vp['cmeans'].set_color('k')

    # axis labels
    ax[ii+3].set_ylabel(label)
    if ii == 1:
        ax[ii+3].set_xlabel('number of non-degenerate inputs (n)')
        ax[ii+3].set_xticks(group_centers)
        ax[ii+3].set_xticklabels(ns)
    ax[ii+3].set_ylim([-0.02,1.02])

# add legend for depth colors
depth_handles = [
    plt.Line2D([0], [0], color='C'+str(d), lw=5, label=f'{d}')
    for d in range(max_depth + 1)
]
a=fig.legend(handles=depth_handles, loc='center', ncol=7, frameon=False,bbox_to_anchor=[0.5,0.92],
             title='canalizing depth of non-degenerate Boolean function' if EXACT_DEPTH else 'minimal canalizing depth')
plt.savefig('all_can_with_bar.pdf',bbox_inches='tight')









width = 0.28
violinplot_args = {'widths': width, 'showmeans': True, 'showextrema': False}

fig = plt.figure(figsize=(7.5,4.5))  # wider to fit two columns nicely
gs  = fig.add_gridspec(
    nrows=6, ncols=2,
    width_ratios=[1, 1],   # columns equally wide
    wspace=0.3, hspace=0.25
)

# Left column: three equally tall axes (each spans 2 rows)
ax2 = fig.add_subplot(gs[4:6, 0])
ax0 = fig.add_subplot(gs[0:2, 0])
ax1 = fig.add_subplot(gs[2:4, 0])

# Right column: two axes, each 1.5× taller (each spans 3 rows)
ax4 = fig.add_subplot(gs[3:6, 1])
ax3 = fig.add_subplot(gs[0:3, 1])


# Keep the rest of your code the same, but replace `ax[...]` with this list:
ax = [ax0, ax1, ax2, ax3, ax4]

# (Your existing code continues below unchanged)
base_gap = 1     # gap between groups
intra_gap = 0.3  # gap within group
max_depth = max(ns)

ax[0].spines[['right', 'top']].set_visible(False)
ax[1].spines[['right', 'top']].set_visible(False)
ax[2].spines[['right', 'top']].set_visible(False)

positions = []
values_expected = []
values_observed = []
colors_used = []
group_centers = []

ax[0].set_xticks([])
ax[1].set_xticks([])
ax[3].set_xticks([])

current_x = 0.0
for i, n in enumerate(ns):
    valid_depths = np.append(np.arange(n-1), n)
    n_viols = len(valid_depths)

    offsets = np.linspace(
        -(n_viols - 1) * intra_gap / 2,
        (n_viols - 1) * intra_gap / 2,
        n_viols
    )
    group_positions = current_x + offsets
    positions.extend(group_positions)
    group_centers.append(current_x)

    for depth in valid_depths:
        values_expected.append(proportion_canalizing_depths[n, depth])
        values_observed.append(observed_proportion_canalizing_depths[n, depth])
        colors_used.append('C' + str(depth))

    group_width = (n_viols - 1) * intra_gap
    current_x += group_width / 2 + base_gap + width + intra_gap

#ax[1].set_xticks(group_centers)
ax[2].set_xticks(group_centers)
ax[2].set_xticklabels(ns)

for vpos, val, c in zip(positions, values_observed, colors_used):
    ax[0].bar([vpos], val, color=c, width=width)
for vpos, val, c in zip(positions, values_expected, colors_used):
    ax[1].bar([vpos], val, color=c, width=width)
    ax[2].bar([vpos], val, color=c, width=width)

ax[2].set_yscale('log')
ax[1].set_ylabel('expected\nproportion')
ax[2].set_ylabel('expected\nproportion [log]')
ax[2].set_yticks([1.e-06, 1.e-04, 1.e-02, 1])
ax[2].set_yticklabels(['$\\mathdefault{10^{-6}}$','$\\mathdefault{10^{-4}}$','$\\mathdefault{10^{-2}}$','1'])
ax[0].set_ylabel('observed\nproportion')
ax[2].set_xlabel('number of non-degenerate inputs (n)')

for ii, (data, label) in enumerate(zip(
    [canalizing_strengths, input_redundancies],
    ['canalizing strength', 'normalized\ninput redundancy'])):

    ax[ii+3].spines[['right', 'top']].set_visible(False)

    positions = []
    values = []
    colors_used = []
    group_centers = []

    current_x = 0.0
    for i, n in enumerate(ns):
        valid_depths = np.append(np.arange(n-1), n)
        n_viols = len(valid_depths)

        offsets = np.linspace(
            -(n_viols - 1) * intra_gap / 2,
            (n_viols - 1) * intra_gap / 2,
            n_viols
        )
        group_positions = current_x + offsets
        positions.extend(group_positions)
        group_centers.append(current_x)

        for depth in valid_depths:
            values.append(data[i, depth, :])
            colors_used.append('C' + str(depth))

        group_width = (n_viols - 1) * intra_gap
        current_x += group_width / 2 + base_gap + width + intra_gap

    for vpos, val, c in zip(positions, values, colors_used):
        vp = ax[ii+3].violinplot(val, positions=[vpos], **violinplot_args)
        for body in vp['bodies']:
            body.set_facecolor(c)
            body.set_alpha(0.85)
        vp['cmeans'].set_color('k')

    ax[ii+3].set_ylabel(label)
    if ii == 1:
        ax[ii+3].set_xlabel('number of non-degenerate inputs (n)')
        ax[ii+3].set_xticks(group_centers)
        ax[ii+3].set_xticklabels(ns)
    ax[ii+3].set_ylim([-0.02, 1.02])


depth_handles = [
    plt.Line2D([0], [0], color='C' + str(d), lw=5, label=f'{d}')
    for d in range(max_depth + 1)
]
a = fig.legend(
    handles=depth_handles, loc='center', ncol=7, frameon=False,
    bbox_to_anchor=[0.5, 0.93],
    title='canalizing depth of non-degenerate Boolean function' if EXACT_DEPTH
          else 'minimal canalizing depth'
)

plt.savefig('all_can_with_bar_2col.pdf', bbox_inches='tight')
