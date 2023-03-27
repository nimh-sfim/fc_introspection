import pandas as pd
import numpy as np
import hvplot.pandas
import holoviews as hv
from holoviews import dim, opts
from nilearn.plotting import plot_matrix
from matplotlib import patches
import matplotlib.pyplot as plt
import networkx as nx
from bokeh.models import FixedTicker
from nxviz.utils import node_table, edge_table
from nxviz import plots, nodes, edges, lines
import os.path as osp
nw_color_map = {'LH-Vis':'#9e53a9',      'RH-Vis':'#9e53a9','Vis':'#9e53a9',
                   'LH-SomMot':'#7c99be'  ,'RH-SomMot':'#7c99be','SomMot':'#7c99be',
                   'LH-DorsAttn':'#439639'    ,'RH-DorsAttn':'#439639','DorsAttn':'#439639',
                   'LH-SalVentAttn':'#da69f9','RH-SalVentAttn':'#da69f9','SalVentAttn':'#da69f9',
                   'LH-Cont':'#eeba42','RH-Cont':'#eeba42','Cont':'#eeba42',
                   'LH-Default':'#db707e','RH-Default':'#db707e','Default':'#db707e',
                   'LH-Subcortical':'#bcbd22',      'RH-Subcortical':'#bcbd22','Subcortical':'#bcbd22',
                   'LH-Limbic':'#f6fdcb',      'RH-Limbic':'#f6fdcb','Limbic':'#f6fdcb'}
hm_color_map = {'LH':'grey','RH':'darkgrey'}

# ========================================================
#       Plotting of annotated FC matrix (full view)
# ========================================================

# This function is used to extract the location of labels and segments when showing the data
# organized by hemispheric membership.
def get_net_divisions_by_Hemisphere(roi_info_input, verbose=False):
    """
    INFO: This function takes as input ROI information and returns the label and location of network
    ===== and hemisphere tickmarks. It also returns the start and end points of colored segments to 
          annotate FC matrices with network and hemisphere information.
          
    INPUTS:
    =======
    roi_info_input: this can be either a string, a pandas dataframe or a pandas multiindex.
       * If string, then assume it is the path to a pandas dataframe with ROI info.
       * If pd.MultiIndex, assume it contains at least three levels labeled Hemisphere, Network and ROI_ID
       * If pd.DataFrame, assume it contains at least three columns labeled Hemisphere, Network and ROI_ID
    """
  
    # Load ROI Info File
    if isinstance(roi_info_input,str):
        if osp.exists(roi_info_input):
            if verbose:
                print('++ INFO [get_net_divisions]: Loading ROI information from disk [%s]' % roi_info_input)
            roi_info = pd.read_csv(roi_info_input)
        else:
            print('++ ERROR [get_net_divisions]: Provided path to ROI information not found [%s]' % roi_info_input)
            return None
    if isinstance(roi_info_input,pd.MultiIndex):
        roi_info = pd.DataFrame()
        roi_info['Hemisphere'] = list(roi_info_input.get_level_values('Hemisphere'))
        roi_info['Network']    = list(roi_info_input.get_level_values('Network'))
        roi_info['ROI_ID']     = list(roi_info_input.get_level_values('ROI_ID'))

    Nrois         = roi_info.shape[0]                                # Total number of ROIs    
    Nrois_LH      = roi_info[roi_info['Hemisphere']=='LH'].shape[0]  # Number of ROIs in LH
    Nrois_RH      = roi_info[roi_info['Hemisphere']=='RH'].shape[0]  # Number of ROIs in RH
    Nnet_segments = len(list(set([row['Hemisphere']+'_'+row['Network'] for r,row in roi_info.iterrows()]))) # Number of NW segments (both hemispheres)
    Nnetworks     = len(roi_info['Network'].unique())                # Number of unique networks (independent of hemisphere)
    net_names     = list(roi_info['Network'].unique())               # Network names (no hm info attached)
    hm_net_names  = list((roi_info['Hemisphere']+'-'+roi_info['Network']).unique()) # Network names (hm info attached)
    if verbose:
        print('++ INFO: Number of ROIs [Total=%d, LH=%d, RH=%d]' % (Nrois,Nrois_LH,Nrois_RH))
        print('++ INFO: Number of Networks [#Nets = %d | #Segments = %d]' % (Nnet_segments,Nnetworks))
        print('++ INFO: Network Names %s' % str(net_names))
        print('++ INFO: Network/Hemi Names %s' % str(hm_net_names))
    
    # Get Positions of start and end of ROIs that belong to the different networks. We will use this info to set labels
    # on matrix axes
    hm_net_edges = [0]
    for network in net_names:
        hm_net_edges.append(roi_info[(roi_info['Network']==network) & (roi_info['Hemisphere']=='LH')].iloc[-1]['ROI_ID'])
    for network in net_names:
        hm_net_edges.append(roi_info[(roi_info['Network']==network) & (roi_info['Hemisphere']=='RH')].iloc[-1]['ROI_ID'])
    net_meds = [int(i) for i in hm_net_edges[0:-1] + np.diff(hm_net_edges)/2]
    if verbose:
        print('++ INFO: Network End      IDs: %s ' % str(hm_net_edges))
        print('++ INFO: Network Midpoint IDs: %s ' % str(net_meds))
    return Nrois, Nrois_LH, Nrois_RH, Nnet_segments, Nnetworks, net_names, hm_net_names, hm_net_edges, net_meds

# This function is used to extract the location of labels and segments when showing the data
# organized by network membership.
def get_net_divisions_by_Network(roi_info_input, verbose=False):
    """
    INFO: This function takes as input ROI information and returns the label and location of network
    ===== and tickmarks. It also returns the start and end points of colored segments to 
          annotate FC matrices with network information.
          
    INPUTS:
    =======
    roi_info_input: this can be either a string, a pandas dataframe or a pandas multiindex.
       * If string, then assume it is the path to a pandas dataframe with ROI info.
       * If pd.MultiIndex, assume it contains at least three levels labeled Hemisphere, Network and ROI_ID
       * If pd.DataFrame, assume it contains at least three columns labeled Hemisphere, Network and ROI_ID
    """
    # Load ROI Info File
    if isinstance(roi_info_input,str):
        if osp.exists(roi_info_input):
            if verbose:
                print('++ INFO [get_net_divisions]: Loading ROI information from disk [%s]' % roi_info_input)
            roi_info = pd.read_csv(roi_info_input)
        else:
            print('++ ERROR [get_net_divisions]: Provided path to ROI information not found [%s]' % roi_info_input)
            return None
    if isinstance(roi_info_input,pd.MultiIndex):
        roi_info = pd.DataFrame()
        roi_info['Hemisphere'] = list(roi_info_input.get_level_values('Hemisphere'))
        roi_info['Network']    = list(roi_info_input.get_level_values('Network'))
        roi_info['ROI_ID']     = np.arange(roi_info.shape[0])+1
        
    Nrois         = roi_info.shape[0]                                # Total number of ROIs
    Nnetworks     = len(roi_info['Network'].unique())                # Number of unique networks (independent of hemisphere)
    net_names     = list(roi_info['Network'].unique())               # Network names (no hm info attached)
    if verbose:
        print('++ INFO: Network Names %s' % str(net_names))
        
    # Get Positions of start and end of ROIs that belong to the different networks. We will use this info to set labels
    # on matrix axes
    net_edges = [0]
    for network in net_names:
        net_edges.append(roi_info[(roi_info['Network']==network)].iloc[-1]['ROI_ID'])
    net_meds = [int(i) for i in net_edges[0:-1] + np.diff(net_edges)/2]
    if verbose:
        print('++ INFO: Network End      IDs: %s ' % str(net_edges))
        print('++ INFO: Network Midpoint IDs: %s ' % str(net_meds))
    return Nrois, Nnetworks, net_names, net_edges, net_meds


def hvplot_fc(data, roi_info_input = None, by='Hemisphere', 
              hm_cmap=hm_color_map, net_cmap=nw_color_map, cbar_title='',
              clim=(-.8,.8), cbar_title_fontsize=16, 
              add_labels=True,add_color_segments=True,
              verbose=False, cmap=['blue','white','red'], nw_sep_lw=0.5, nw_sep_ld='dashed'):
    """
    INFO: This function will generate an annotated and interactive view of a given FC matrix.
    
    """
    # If ROI information is not provided explicitly, it is assumed that it is included in the index and columns of the input data structure
    # -------------------------------------------------------------------------------------------------------------------------------------
    if roi_info_input is None:
        roi_info_input = data.index
    
    # Get information needed for correct labeling of the matrix
    # ---------------------------------------------------------
    assert by in ['Hemisphere','Network']
    if by == 'Hemisphere':
        Nrois, Nrois_LH, Nrois_RH, Nnet_segments, Nnetworks, net_names, hm_net_names, hm_net_edges, hm_net_meds = get_net_divisions_by_Hemisphere(roi_info_input)
    if by == 'Network':
        Nrois, Nnetworks, net_names, net_edges, net_meds = get_net_divisions_by_Network(roi_info_input)
    
    # Remove axes from data
    # ---------------------
    if isinstance(data,pd.DataFrame):
        if verbose:
            print('++ INFO[hvplot_fc]: removing index and column names from inputted data structure')
        data = data.values
    matrix_to_plot              = pd.DataFrame(data)
    matrix_to_plot.index        = np.arange(matrix_to_plot.shape[0])
    matrix_to_plot.index.name   = 'ROI2'
    matrix_to_plot.columns      = np.arange(matrix_to_plot.shape[1])
    matrix_to_plot.columns.name = 'ROI1'

    # Create Y axis ticks and labels
    # ------------------------------
    if by == 'Hemisphere':
        y_ticks = hm_net_meds + list (np.array(hm_net_meds) + Nrois_LH)
        y_tick_labels = net_names + net_names
        y_ticks_info = list(tuple(zip(y_ticks, y_tick_labels)))
    
        x_ticks       = [Nrois_LH/2, Nrois_LH + (Nrois_RH/2)]
        x_tick_labels = ['Left Hemisphere','Right Hemisphere']
        x_ticks_info  = list(tuple(zip(x_ticks, x_tick_labels)))
        x_rotation    = 0
    if by == 'Network':
        y_ticks = net_meds 
        y_tick_labels = net_names 
        y_ticks_info = list(tuple(zip(y_ticks, y_tick_labels)))
        x_ticks, x_ticks_info = y_ticks, y_ticks_info
        x_rotation   = 90
    
    # Create X-axis color segment bar if needed
    # -----------------------------------------
    if add_color_segments & (by=='Hemisphere'):
        color_segments_x = hv.Segments(((-1,Nrois_LH),(-2.5,-2.5),(Nrois_LH,Nrois),(-2.5,-2.5),('LH','RH')), vdims='Hemispheres').opts(cmap=hm_color_map, color=dim('Hemispheres'), line_width=10,show_legend=False,xrotation=x_rotation)
        y_min_lim = -4
    elif add_color_segments & (by == 'Network'):
        color_segments_x = hv.Segments((tuple(np.array(net_edges[:-1])-0.5),
                                        tuple(-2.5*np.ones(Nnetworks)),
                                        tuple(np.array(net_edges[1:])-0.5),
                                        tuple(-2.5*np.ones(Nnetworks)), net_names),vdims='Networks').opts(cmap=nw_color_map, color=dim('Networks'), line_width=10, show_legend=False, xrotation=x_rotation)
        y_min_lim = -4
    else:
        color_segments_x = None
        y_min_lim = 0
        
    # Create Y-axis color segment bar if needed
    # -----------------------------------------
    if add_color_segments & (by=='Hemisphere'):
        color_segments_y = hv.Segments((tuple(-2.5*np.ones(Nnet_segments)),tuple(np.array(hm_net_edges[:-1])-0.5),
                              tuple(-2.5*np.ones(Nnet_segments)),tuple(np.array(hm_net_edges[1:])-0.5), hm_net_names),vdims='Networks').opts(cmap=nw_color_map, color=dim('Networks'), line_width=10,show_legend=False)
        x_min_lim = -4
    elif add_color_segments & (by=='Network'):
        color_segments_y = hv.Segments((tuple(-2.5*np.ones(Nnetworks)),tuple(np.array(net_edges[:-1])-0.5),
                              tuple(-2.5*np.ones(Nnetworks)),tuple(np.array(net_edges[1:])-0.5), net_names),vdims='Networks').opts(cmap=nw_color_map, color=dim('Networks'), line_width=10,show_legend=False)
        x_min_lim = -4
    else:
        color_segments_y = None
        x_min_lim = 0
        
    # Create Heatmap with or without text labels
    # ------------------------------------------
    dict_heatmapopts = dict(color_levels=[-1,-0.10,0.10,1], cmap=cmap)
    if add_labels:
        plot           = matrix_to_plot.hvplot.heatmap(aspect='square', frame_width=500 , cmap=cmap,
                                                       clim=clim, xlim=(x_min_lim,Nrois-.5), ylim=(y_min_lim,Nrois-.5), 
                                                       yticks=y_ticks_info, xticks= x_ticks_info, 
                                                       fontsize={'ticks':12,'clabel':cbar_title_fontsize}).opts(xrotation=x_rotation, colorbar_opts={'title':cbar_title, 
                                                                                                                               'major_label_overrides':{-0.5:'F2 > F1',0:'',0.5:'F1 > F2'}, 
                                                                                                                               'ticker': FixedTicker(ticks=[-1.5,-0.5,0.5,1.5])}, **dict_heatmapopts)

    else:
        plot           = matrix_to_plot.hvplot.heatmap(aspect='square', frame_width=500 , cmap=cmap,
                                                       clim=clim, xlim=(x_min_lim,Nrois-.5), 
                                                       ylim=(y_min_lim,Nrois-.5), 
                                                       yaxis=None, xaxis=None,
                                                       fontsize={'ticks':12,'clabel':cbar_title_fontsize}).opts(xrotation=x_rotation, colorbar_opts={'title':cbar_title, 
                                                                                                                               'major_label_overrides':{-0.5:'F2 > F1',0:'',0.5:'F1 > F2'}, 
                                                                                                                               'ticker': FixedTicker(ticks=[-1.5,-0.5,0.5,1.5])}, **dict_heatmapopts)
    
    # Add Line Separation Annotations
    # -------------------------------
    if by == 'Hemisphere':
        for x in hm_net_edges:
            plot = plot * hv.HLine(x-.5).opts(line_color='k',line_dash=nw_sep_ld, line_width=nw_sep_lw)
            plot = plot * hv.VLine(x-.5).opts(line_color='k',line_dash=nw_sep_ld, line_width=nw_sep_lw)
    if by == 'Network':
        for x in net_edges:
            plot = plot * hv.HLine(x-.5).opts(line_color='k',line_dash=nw_sep_ld, line_width=nw_sep_lw)
            plot = plot * hv.VLine(x-.5).opts(line_color='k',line_dash=nw_sep_ld, line_width=nw_sep_lw)
    
    # Add Side Colored Segments if instructed to do so
    # ------------------------------------------------
    plot = plot * hv.HLine(y_min_lim).opts(line_color='k',line_dash='solid', line_width=2)
    plot = plot * hv.VLine(x_min_lim).opts(line_color='k',line_dash='solid', line_width=2)
    plot = plot * hv.HLine(x-.5).opts(line_color='k',line_dash='solid', line_width=2)
    plot = plot * hv.VLine(x-.5).opts(line_color='k',line_dash='solid', line_width=2)
    for segment in [color_segments_y, color_segments_x]:
        if segment is not None:
            plot = plot * segment
    return plot
   


def plot_fc(data,roi_info_path, hm_cmap=hm_color_map, net_cmap=nw_color_map, cbar_title='',title='',
                    clim=(-.8,.8), cbar_title_fontsize=16, 
                    add_net_colors=True, add_net_labels=True,
                    add_hm_colors=True, add_hm_labels=True, verbose=False, figsize=(5,5)):
    # Get network and hemisphere information
    Nrois, Nrois_LH, Nrois_RH, Nnet_segments, Nnetworks, net_names, hm_net_names, hm_net_borders, hm_net_meds = get_net_divisions(roi_info_path, verbose=verbose)
    
    # Plot heatmap
    fig, ax = plt.subplots(1,1,figsize=figsize)
    plot = plot_matrix(data, vmin=clim[0], vmax=clim[1], axes=ax)
    plt.title(title,fontsize=14,loc='left')

    # Add line dividers per network
    for x in hm_net_borders[1::]:
        ax.plot([x-.5,x-.5],[0,Nrois],'k--',lw=1)
        ax.plot([-4,Nrois],[x-.5,x-.5],'k--',lw=1)
    # Add vertical line separating color-codes from matrix in y axis
    ax.plot([0,0],[0,Nrois],'k',lw=1)

    # Add color coded network blockx
    hm_net_inits = hm_net_borders[:-1]
    hm_net_ends  = hm_net_borders[1::]

    for hm_net,start_point,end_point,mid_point in zip(hm_net_names, hm_net_inits,hm_net_ends, hm_net_meds):
        aux_patch = patches.Rectangle((-4,start_point-.5),4,end_point, edgecolor='none',facecolor=nw_color_map[hm_net])
        ax.add_patch(aux_patch)

    # Add color coded hemisphere blocks
    lh_patch = patches.Rectangle((0,-4),Nrois_LH,4, edgecolor='none', facecolor=hm_color_map['LH'])
    rh_patch = patches.Rectangle((Nrois_LH,-4),Nrois_RH,4, edgecolor='none', facecolor=hm_color_map['RH'])
    ax.add_patch(lh_patch)
    ax.add_patch(rh_patch)
    ax.plot([0,Nrois],[0,0],'k',lw=1)

    # Flip for consistency
    ax.set_ylim(-4,Nrois-1)
    ax.set_xlim(-4,Nrois-1)
    # Add network labels
    ax.set_yticks(hm_net_meds)
    ax.set_yticklabels([item.split('-')[1] for item in hm_net_names], fontsize=12);
    # Add hemisphere labels
    ax.set_xticks([Nrois_LH/2, Nrois_LH + (Nrois_RH/2)])
    ax.set_xticklabels(['Left Hemisphere','Right Hemisphere'], fontsize=12);
    
    plt.close()
    return fig
   
def hvplot_fc_nwlevel(data,mode='percent',clim_max=None,clim_min=0, cmap='viridis', title='', add_net_colors=False):
    """
    This function plots a summary view of how many within- and between- network connections
    are significantly different in a given contrast.
    
    Inputs:
    data: this is dataframe with the outputs from NBS or a combination of those. It is expected to only contains 0,1 and -1 values.
          it is also expected to be indexed by a multi-index with one level having the name 'Network' that contains the name of the
          networks. This is used for plotting and counting purposes, so it is very important to adhere to this requirement.
    
    mode: instruct to report results in terms of absolute number of connections 'count' or percentage of possible connections 'percent'.
    
    clim_max: max value for the colorbar. If unset, it will be automatically set to the 95% percentile.
    
    clim_min: min value for the colorbar. It unset, it will be automatically set to zero.
    
    cmap: colormap. [default = viridis]

    title: title for the plot. [default = '']
    
    add_net_colors: flag to remove substitute text labels in the X-axis by colored segments denoting the different networks.
    """
    assert mode in ['percent','count']
    data         = data.copy()
    networks     = list(data.index.get_level_values('Network').unique())
    num_networks = len(networks)
    num_sig_cons = pd.DataFrame(index=networks, columns=networks)
    pc_sig_cons  = pd.DataFrame(index=networks, columns=networks)
    for n1 in networks:
        for n2 in networks:
            aux = data.loc[data.index.get_level_values('Network')==n1,data.columns.get_level_values('Network')==n2]
            if n1 == n2:
                ncons = aux.shape[0] * (aux.shape[0] -1 ) / 2
                num_sig_cons.loc[n1,n2] = int(aux.sum().sum()/2) 
                pc_sig_cons.loc[n1,n2]  = 100 * num_sig_cons.loc[n1,n2] / ncons
            else:
                ncons = aux.shape[0] * aux.shape[1] 
                num_sig_cons.loc[n1,n2] = int(aux.sum().sum())
                pc_sig_cons.loc[n1,n2]  = 100 * num_sig_cons.loc[n1,n2] / ncons
    num_sig_cons = num_sig_cons.infer_objects()
    pc_sig_cons  = pc_sig_cons.infer_objects()
    #Advance plotting mode with colored segments in the horizontal axis.
    if add_net_colors:
       # Create Y axis ticks and labels
       y_ticks_info = list(tuple(zip(range(num_networks), networks))) 
       # Create Network Colorbar
       if add_net_colors:
           net_segments_y = hv.Segments((tuple(np.arange(num_networks)+1),tuple(np.ones(num_networks)-1.5),
                                      tuple(np.arange(num_networks)),tuple(np.ones(num_networks)-1.5), networks),vdims='Networks').opts(cmap=nw_color_map, color=dim('Networks'), line_width=10,show_legend=False)    
    
       # Remove axes from data
       if mode=='percent':
           matrix_to_plot              = pd.DataFrame(pc_sig_cons.values)
           cbar_title = 'Percent of Connections:'
       else:
           matrix_to_plot              = pd.DataFrame(num_sig_cons.values)
           cbar_title = 'Number of Connections:'
       matrix_to_plot.index        = np.arange(matrix_to_plot.shape[0], dtype=int)
       matrix_to_plot.columns      = np.arange(matrix_to_plot.shape[1], dtype=int)
       #return matrix_to_plot 
       if clim_max is None:
          clim_max = matrix_to_plot.quantile(.95).max()
       heatmap = matrix_to_plot.round(1).hvplot.heatmap(aspect='square', clim=(0,clim_max), cmap=cmap, 
                                                        title=title, ylim=(-.5,num_networks-.5), yticks= y_ticks_info, xaxis=None).opts(xrotation=90,colorbar_opts={'title':cbar_title})
       plot = heatmap * hv.Labels(heatmap).opts(opts.Labels(text_color='white'))
       plot = plot * net_segments_y
    
    #Basic Plotting Mode
    else:
        if mode=='percent':
            if clim_max is None:
                clim_max = pc_sig_cons.quantile(.95).max()
            heatmap = pc_sig_cons.round(1).hvplot.heatmap(aspect='square', clim=(0,clim_max), cmap=cmap, title=title).opts(xrotation=90, colorbar_opts={'title':'Percent of Connections:'})
            plot = heatmap * hv.Labels(heatmap).opts(opts.Labels(text_color='white')) 
        if mode=='count':
            if clim_max is None:
                clim_max = num_sig_cons.quantile(.95).max()
            heatmap  = num_sig_cons.round(1).hvplot.heatmap(aspect='square', clim=(clim_min,clim_max), cmap=cmap, title=title ).opts(xrotation=90,colorbar_opts={'title':'Number of Connections:'})
            plot = heatmap * hv.Labels(heatmap).opts(opts.Labels(text_color='white'))
    #Return plot
    return plot   

# CIRCOS PLOTS
# ============
def get_node_positions(roi_info,r=0.5,c_x=0.0,c_y=0.0,hemi_gap=5):
    # Get list of networks
    # ====================
    networks    = list(roi_info['Network'].unique())
    # Create Node Positions
    # =====================
    # 1. Max number of nodes per networks in one or another hemisphere
    max_num_nodes_per_network = {}
    for nw in networks:
        aux = roi_info[roi_info['Network']==nw]
        max_num_nodes_per_network[nw] = np.max([aux[aux['Hemisphere']=='LH'].shape[0],aux[aux['Hemisphere']=='RH'].shape[0]])
    # 2. Number of locations to choose from in circle: 2 * (Max_PerNetwork + Pads on both ends)
    n_locations = 2 * (np.array([max_num_nodes_per_network[nw] for nw in max_num_nodes_per_network.keys()]).sum() + 2 * hemi_gap)
    
    # 3. Evently distributed angels (in radians) across the circle
    avail_angles = np.linspace(0,2*np.pi,n_locations)
    avail_angles = np.roll(avail_angles,-int(n_locations/4))
    
    # 4. Convert the angles to (x,y) locations
    positions = []
    for t in avail_angles:
        x = r*np.cos(t) + c_x
        y = r*np.sin(t) + c_y
        positions.append(np.array([x,y]))
    positions_per_hemi = {'RH': positions[0:int(n_locations/2)], 
                          'LH': positions[int(n_locations/2)::][::-1]}
    # 5. Create Final Layout Object
    G_circos_layout = {}
    for hm in ['LH','RH']:
        index = hemi_gap
        for nw in networks:
            index_init = index 
            aux = roi_info[(roi_info['Network']==nw) & (roi_info['Hemisphere']==hm)]
            for n,node in enumerate(list(aux['ROI_ID'])):
                G_circos_layout[node] = positions_per_hemi[hm][index]
                index = index + 1
            index = index_init + max_num_nodes_per_network[nw]
    return G_circos_layout

def plot_as_circos(data,roi_info,figsize=(10,10),edge_weight=2,title=None, hemi_gap=5, show_pos=True, show_neg=True):
    # Check inputs meet expectations
    assert isinstance(data,pd.DataFrame), "++ERROR [plot_as_circos]: input data expected to be a pandas dataframe"
    assert 'ROI_ID'     in data.index.names, "++ERROR [plot_as_circos]: roi_info expected to have one column named ROI_ID"
    assert 'Hemisphere' in roi_info.columns, "++ERROR [plot_as_circos]: roi_info expected to have one column named Hemisphere"
    assert 'Network'    in roi_info.columns, "++ERROR [plot_as_circos]: roi_info expected to have one column named Network"
    assert 'RGB'    in roi_info.columns, "++ERROR [plot_as_circos]: roi_info expected to have one column named RGB"

    # Convert input to ints (this function only works for unweigthed graphs)
    fdata          = data.copy().astype('int')
    fdata.index    = fdata.index.get_level_values('ROI_ID')
    fdata.columns  = fdata.index
    # Create None objects for the graphs
    G_pos, G_neg, G = None, None, None
    # Count the input values
    val_counts = pd.Series(fdata.values.flatten()).value_counts()
    # Check for the presence of positive edges
    if 1 in val_counts.index:
        fdata_pos              = fdata.copy()
        #fdata_pos.index        = fdata.index.get_level_values('ROI_ID')
        #fdata_pos.columns      = fdata.index.get_level_values('ROI_ID')
        fdata_pos[fdata_pos<0] = 0 # Removing -1 from positive graph
        G_pos = nx.from_pandas_adjacency(fdata_pos)
    # Check for the present of negative edges
    if -1 in val_counts.index:
        fdata_neg              = fdata.copy()
        #fdata_neg.index        = fdata.index.get_level_values('ROI_ID')
        #fdata_neg.columns      = fdata.index.get_level_values('ROI_ID')
        fdata_neg[fdata_neg>0] = 0 # Removing 1 from negative graph
        G_neg = nx.from_pandas_adjacency(fdata_neg)
    # Create Graph with all nodes (for positioning purposes - edges do not matter)
    
    G             = nx.from_pandas_adjacency(fdata+100) # Ensure we have a graph with all nodes
    # Add information about positive or negative edge
    model_attribs = {}
    if G_pos is not None:
        for edge in G_pos.edges:
            model_attribs[edge] = 'pos'
    if G_neg is not None:
        for edge in G_neg.edges:
            model_attribs[edge] = 'neg'
    nx.set_edge_attributes(G,model_attribs,'Model')
    # Add information about hemisphere and network
    hemi_attribs = {row['ROI_ID']:row['Hemisphere'] for r,row in roi_info.iterrows()}
    nw_attribs   = {row['ROI_ID']:row['Network'] for r,row in roi_info.iterrows()}
    nx.set_node_attributes(G,hemi_attribs,'Hemi')
    nx.set_node_attributes(G,nw_attribs,'Network')
    # Obtain Node Positions
    pos =  get_node_positions(roi_info, hemi_gap=hemi_gap)
    # Node Styling
    nt  = node_table(G)
    node_color = roi_info.set_index('ROI_ID')['RGB']
    node_alpha = pd.Series(1.0, index=node_color.index)
    node_size  = pd.Series(0.01, index=node_color.index)
    # Positive Edges Styling
    if G_pos is not None:
        pos_et         = edge_table(G_pos)
        pos_et_color   = pd.Series('red', index=range(pos_et.shape[0]))
        pos_lw         = edge_weight*pos_et["weight"] 
        pos_alpha      = pd.Series(0.5, index=range(pos_et.shape[0]))
    # Negative Edges Styling
    if G_neg is not None:
        neg_et         = edge_table(G_neg)
        neg_et_color   = pd.Series('lightblue', index=range(neg_et.shape[0]))#edges.edge_colors(et, nt=None, color_by=None, node_color_by=None)
        neg_lw         = edge_weight*neg_et["weight"] 
        neg_alpha      = pd.Series(0.3, index=range(neg_et.shape[0]))
    #Create plot
    fig, ax = plt.subplots(1,1,figsize=figsize)
    patches = nodes.node_glyphs( nt, pos, node_color=node_color, alpha=node_alpha, size=node_size, **{'edgecolor':None, 'linewidth':0})
    for patch in patches: 
        ax.add_patch(patch)
    if (G_pos is not None) & (show_pos is True): 
        patches = lines.circos( pos_et, pos, edge_color=pos_et_color, alpha=pos_alpha, lw=pos_lw, aes_kw={"fc": "none"} ) 
        for patch in patches: 
            ax.add_patch(patch)
    if (G_neg is not None) & (show_neg is True):
        patches = lines.circos( neg_et, pos, edge_color=neg_et_color, alpha=neg_alpha, lw=neg_lw, aes_kw={"fc": "none"} ) 
        for patch in patches: 
            ax.add_patch(patch)
    if title is not None:
       plt.title(title)
    plots.rescale(G) 
    plots.aspect_equal()
    plots.despine()
    plt.close()
    return fig