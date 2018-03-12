import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.gridspec as gridspec

from moviepy.video.io.bindings import mplfig_to_npimage
import  moviepy.editor as mpy
import seaborn as sns
from scipy.stats import entropy

sns.set()


#converts decimals to binary
def int_to_binary(num, pad=4):
    return bin(num)[2:].zfill(pad)


# Computes the hamming distance between two genotypes.
def hammingDistance(s1, s2):
    assert len(s1) == len(s2)
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

# Converts an integer to a genotype by taking the binary value and padding to the left by 0s
def convertIntToGenotype(anInt, pad):
	offset = 2**pad
	return [int(x) for x in bin(offset+anInt)[3:]]

#Generates transition matrix for random mutations (deleterious mutations included)
def random_mutations(N):
    trans_mat = np.zeros([N,N])
    for mm in range(N):
        for nn in range(N):
            trans_mat[mm, nn] = hammingDistance( int_to_binary(mm) , int_to_binary(nn))

    trans_mat[trans_mat>1] = 0
    trans_mat = trans_mat/trans_mat.sum(axis=1)

    return trans_mat

#Clean up average growth rates data
def clean_up_data():
    landscapes = pd.read_csv('../data/avg_growth_rates.csv').dropna(how='all', axis=1).dropna(how='any', axis=0)

    cols = list(landscapes.columns)

    for mm, col in enumerate(cols):
        if mm>0 and mm<17:
            cols[mm] =col.replace(' ', '').zfill(4)

    landscapes = landscapes.rename(columns=dict(zip(list(landscapes.columns), cols)))
    sorted_cols = cols[:1]+list(np.sort(cols[1:17]))+cols[17:]
    landscapes = landscapes.reindex_axis(sorted_cols, axis=1)
    landscapes.to_csv('../clean_data.csv', index=False)


#Run the automaton
#Implements cell division. The division rates are based on the experimental data
def run_automaton(fit_land, #Fitness landscape
                n_gen = 40, #Number of simulated generations
                mut_rate = 0.01, #probability of mutation per generation
                max_cells = 10**5, # Max number of cells
                death_rate = 0.05 # Death rate
                ):

    #Obtain transition matrix for mutations
    P = random_mutations(len(fit_land))

    #Number of different alleles
    n_allele = len(P)

    #initial number of n_cells, 10 of each type
    n_cells = 10*n_allele

    #This array keeps track of allele type of cells
    cell_types = np.zeros(max_cells).astype('int64')

    #Division rates are determined by the allele type (from experimental data) and tracked in this array
    div_rate = np.zeros(max_cells)
    cell_types[:n_cells] = (n_cells//n_allele)*list( range(n_allele) )

    #experimental growth rates are normalized (divided by the maximum)
    #such that the fittest divided once per generation
    div_rate[:n_cells] = (n_cells//n_allele)*list(fit_land/fit_land.max())

    #Keeps track of cell counts at each generation
    counts = np.zeros([n_gen, n_allele])

    for mm in range(n_gen):

        ### Death of cells
        death_rates = np.random.rand(n_cells)
        indice = np.where( death_rates>death_rate)[0]
        temp = div_rate[indice]
        n_cells = len(temp)
        div_rate[:] = 0
        div_rate[:n_cells] = temp

        temp = cell_types[indice]
        cell_types[:] = 0
        cell_types[:n_cells] = temp

        #Count allele types
        if n_cells>0:
            counts[mm] = np.bincount(cell_types[:n_cells], minlength=n_allele)
        else:
            counts = counts[:mm]
            break

        ### Divide cells based on the division rates
        #Scale division rates based on carrying capacity
        division_scale = 1/ ( 1+(2*n_cells/max_cells)**4)
        division_rates = np.random.rand(n_cells)
        indice = division_rates<division_scale*div_rate[:n_cells]

        #Do not let cell count exceed the max_cells
        new_cells = np.sum(indice)
        if (n_cells+new_cells)>max_cells:
            new_cells = max_cells - n_cells

        #Copy allele type of daughter cells
        cell_types[n_cells:n_cells+new_cells] = cell_types[:n_cells][indice][:new_cells]
        #copy division rate of daughter cells
        div_rate[n_cells:n_cells+new_cells] = div_rate[:n_cells][indice][:new_cells]


        ### Mutate cells
        #initial state of allele types
        init_types = cell_types[n_cells:n_cells+new_cells].copy()

        #Final state of allele types
        final_types = init_types.copy()
        #Generate random numbers to check for mutation
        mutation_rates = np.random.rand(new_cells)

        #Mutate cells of each allele type
        for allele in np.random.permutation( np.arange(n_allele) ):
            indice = np.logical_and( init_types==allele , mutation_rates < mut_rate )
            mutations = np.random.choice(n_allele, size=np.sum(indice), p=P[allele])

            #Change allele type in final_types
            np.putmask(final_types, indice, mutations)

        #Copy final allele type to tracking array
        cell_types[n_cells:n_cells+new_cells] = final_types
        n_cells = n_cells+new_cells

    return P, counts



def plot_hypercube(pop, ax=None, N=4, node_scale = None, base_node_size = 25.):
    """
    Plots the population counts on the hypercube
    """
    #The scaling of node size by population at that node
    if node_scale is None:
        node_scale=500/np.max(pop)
    #Drawing machinary
    G = nx.hypercube_graph(N)
    pos=nx.spectral_layout(G) # positions for all nodes
    labpos = {k : np.array([p[0],p[1]+0.1]) for k,p in pos.items() }
    labels={}
    node_color = []
    node_list  = []
    node_size = []
    if ax is None:
        plt.figure(figsize=[10,10])
        ax= plt.subplot(111)

    #Get color and size for each node
    for i in range(len(pop)):
        labels[tuple(convertIntToGenotype(i,N))] = "".join(map(str,convertIntToGenotype(i,N)))
        node_color.append('r')
        node_size.append( base_node_size+node_scale*pop[i] )
        node_list.append( tuple(convertIntToGenotype(i,N)) )

    nx.draw_networkx_nodes(G, pos, nodelist=node_list,
                            node_color=node_color,
                            node_size=node_size,
                            alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.6, ax=ax)
    nx.draw_networkx_labels(G,labpos,labels,font_size=15, ax=ax)
    return ax

def animate_histogram(counts, title_str=None, num_fps = 10):
    """
    Generates an animation with 4 plots.
    """

    N = len(counts)
    duration = N / num_fps
    n_allele = counts.shape[1]
    x_data = np.arange(n_allele)
    ticks = [int_to_binary(mm) for mm in x_data]
    counts = np.transpose( counts.T/counts.sum(axis=1) )

    #Kullback-Leibler Divergence previous to current
    KL_diver_prev = np.zeros(N)
    #Kullback-Leibler Divergence initial to current
    KL_diver_init = np.zeros(N)
    for nn in range(1,N):
        KL_diver_prev[nn] = entropy(counts[nn-1], counts[nn])
        KL_diver_init[nn] = entropy(counts[0], counts[nn])

    fig = plt.figure(0, figsize=(12,12))
    if title_str is not None:
        plt.suptitle(title_str, fontsize=20)
    gs = gridspec.GridSpec(2, 2, hspace=0.2, wspace=0.2)

    #Plot the population on the hypercube
    ax= fig.add_subplot(gs[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)

    #Plot the population on the histogram
    ax1= fig.add_subplot(gs[1])
    ax1.set_xticks(x_data)
    ax1.set_xticklabels(ticks, rotation=90)
    ax1.set_ylabel('Population ratio')
    ax1.set_ylim([0,1])

    #Kullback-Leibler Divergence previous to current
    ax2= fig.add_subplot(gs[2])
    ax2.set_xlim([0, N])
    ax2.set_ylim([0, np.nanmax(KL_diver_prev)*1.1])
    ax2.set_ylabel('Kullback-Leibler Divergence')
    ax2.set_xlabel('Generation')

    #Kullback-Leibler Divergence initial to current
    ax3= fig.add_subplot(gs[3])
    ax3.set_xlim([0, N])
    ax3.set_ylim([0, np.nanmax(KL_diver_init)*1.1])
    ax3.set_ylabel('Kullback-Leibler Divergence')
    ax3.set_xlabel('Generation')

    node_scale = 400
    #Make frame function for each timepoint
    def make_frame(t, node_scale=node_scale):

        nn = int(t*num_fps)

        plot_hypercube(counts[nn], ax=ax, node_scale=node_scale)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1)

        points0 = ax1.bar(x_data, counts[nn], color='#4682B4')

        ax2.plot( np.arange(nn+1), KL_diver_prev[:nn+1], color='#4682B4')
        ax2.scatter(nn, KL_diver_prev[nn], color='#4682B4', s=20)

        ax3.plot( np.arange(nn+1), KL_diver_init[:nn+1], color='#4682B4')
        ax3.scatter(nn, KL_diver_init[nn], color='#4682B4', s=20)

        output = mplfig_to_npimage( fig )
        points0.remove()
        ax.clear()
        return output

    animation =mpy.VideoClip(make_frame, duration=duration)
    plt.close(0)
    return animation, num_fps
