import pandas as pd
import numpy as np
import hvplot.pandas
import holoviews as hv
from holoviews import dim

nw_color_map = {'LH-Vis':'purple',      'RH-Vis':'purple',
                   'LH-SomMot':'lightblue'  ,'RH-SomMot':'lightblue',
                   'LH-DorsAttn':'green'    ,'RH-DorsAttn':'green',
                   'LH-SalVentAttn':'violet','RH-SalVentAttn':'violet',
                   'LH-Cont':'orange','RH-Cont':'orange',
                   'LH-Default':'red','RH-Default':'red'}
hm_color_map = {'LH':'grey','RH':'darkgrey'}

def hvplot_fc(data,roi_info_path, hm_cmap=hm_color_map, net_cmap=nw_color_map, cbar_title='',
                    clim=(-.8,.8), cbar_title_fontsize=16, 
                    add_net_colors=True, add_net_labels=True,
                    add_hm_colors=True, add_hm_labels=True, verbose=False):
    
    # Load ROI Info File
    roi_info      = pd.read_csv(roi_info_path)
    Nrois         = roi_info.shape[0]
    Nrois_LH      = roi_info[roi_info['Hemisphere']=='LH'].shape[0]
    Nrois_RH      = roi_info[roi_info['Hemisphere']=='RH'].shape[0]
    Nnet_segments = len(list(set([row['Hemisphere']+'_'+row['Network'] for r,row in roi_info.iterrows()])))
    Nnetworks     = len(roi_info['Network'].unique())
    net_names     = list(roi_info['Network'].unique())
    hm_net_names  = list((roi_info['Hemisphere']+'-'+roi_info['Network']).unique())
    if verbose:
        print('++ INFO: Number of ROIs [Total=%d, LH=%d, RH=%d]' % (Nrois,Nrois_LH,Nrois_RH))
        print('++ INFO: Number of Networks [#Nets = %d | #Segments = %d]' % (Nnet_segments,Nnetworks))
        print('++ INFO: Network Names %s' % str(net_names))
        print('++ INFO: Network/Hemi Names %s' % str(hm_net_names))
    
    # Get Positions of start and end of ROIs that belong to the different networks. We will use this info to set labels
    # on matrix axes
    net_ends = [0]
    for network in net_names:
        net_ends.append(roi_info[(roi_info['Network']==network) & (roi_info['Hemisphere']=='LH')].iloc[-1]['ROI_ID'])
    for network in net_names:
        net_ends.append(roi_info[(roi_info['Network']==network) & (roi_info['Hemisphere']=='RH')].iloc[-1]['ROI_ID'])
    net_meds = [int(i) for i in net_ends[0:-1] + np.diff(net_ends)/2]
    if verbose:
        print('++ INFO: Network End      IDs: %s ' % str(net_ends))
        print('++ INFO: Network Midpoint IDs: %s ' % str(net_meds))
    
    # Remove axes from data
    if data is pd.DataFrame:
        data = data.values
    matrix_to_plot              = pd.DataFrame(data)
    matrix_to_plot.index        = np.arange(matrix_to_plot.shape[0])
    matrix_to_plot.index.name   = 'ROI2'
    matrix_to_plot.columns      = np.arange(matrix_to_plot.shape[1])
    matrix_to_plot.columns.name = 'ROI1'

    # Create Y axis ticks and labels
    y_ticks = net_meds + list (np.array(net_meds) + Nrois_LH)
    y_tick_labels = net_names + net_names
    y_ticks_info = list(tuple(zip(y_ticks, y_tick_labels)))
    
    x_ticks       = [Nrois_LH/2, Nrois_LH + (Nrois_RH/2)]
    x_tick_labels = ['Left Hemisphere','Right Hemisphere']
    x_ticks_info  = list(tuple(zip(x_ticks, x_tick_labels)))
    
    # Create Hemisphere Colorbar
    if add_hm_colors:
        hm_segments_x  = hv.Segments(((-1,Nrois_LH),(-2.5,-2.5),(Nrois_LH,Nrois),(-2.5,-2.5),('LH','RH')), vdims='Hemispheres').opts(cmap=hm_color_map, color=dim('Hemispheres'), line_width=10,show_legend=False)
        y_min_lim = -4
    else:
        y_min_lim = 0
    # Create Network Colorbar
    if add_net_colors:
        net_segments_y = hv.Segments((tuple(-2.5*np.ones(Nnet_segments)),tuple(np.array(net_ends[:-1])-0.5),
                              tuple(-2.5*np.ones(Nnet_segments)),tuple(np.array(net_ends[1:])-0.5), hm_net_names),vdims='Networks').opts(cmap=nw_color_map, color=dim('Networks'), line_width=10,show_legend=False)
        x_min_lim = -4
    else:
        x_min_lim = 0
    # Create Heatmap
    if add_hm_labels & add_net_labels:
        plot           = matrix_to_plot.hvplot.heatmap(aspect='square', frame_width=500 , cmap='RdBu_r',
                                                   clim=clim, xlim=(x_min_lim,Nrois-.5), ylim=(y_min_lim,Nrois-.5), 
                                                   yticks=y_ticks_info, xticks= x_ticks_info, fontsize={'ticks':12,'clabel':cbar_title_fontsize}).opts( colorbar_opts={'title':cbar_title})
    if add_hm_labels &  (not add_net_labels):
        plot           = matrix_to_plot.hvplot.heatmap(aspect='square', frame_width=500 , cmap='RdBu_r',
                                                   clim=clim, xlim=(x_min_lim,Nrois-.5), ylim=(y_min_lim,Nrois-.5), yaxis=None,
                                                   xticks= x_ticks_info, fontsize={'ticks':12,'clabel':cbar_title_fontsize}).opts( colorbar_opts={'title':cbar_title})
    if (not add_hm_labels) &  add_net_labels:
        plot           = matrix_to_plot.hvplot.heatmap(aspect='square', frame_width=500 , cmap='RdBu_r',
                                                   clim=clim, xlim=(x_min_lim,Nrois-.5), ylim=(y_min_lim,Nrois-.5), xaxis=None,
                                                   yticks= y_ticks_info, fontsize={'ticks':12,'clabel':cbar_title_fontsize}).opts( colorbar_opts={'title':cbar_title})
    if (not add_hm_labels) & (not add_net_labels):
        plot           = matrix_to_plot.hvplot.heatmap(aspect='square', frame_width=500 , cmap='RdBu_r',
                                                   clim=clim, xlim=(x_min_lim,Nrois-.5), ylim=(y_min_lim,Nrois-.5), yaxis=None, xaxis=None,
                                                   fontsize={'ticks':12,'clabel':cbar_title_fontsize}).opts( colorbar_opts={'title':cbar_title})
    # Add Line Separation Annotations
    for x in net_ends:
        plot = plot * hv.HLine(x-.5).opts(line_color='k',line_dash='solid',line_width=1)
        plot = plot * hv.VLine(x-.5).opts(line_color='k',line_dash='solid',line_width=1)
    # Add Side Colored Segments if instructed to do so
    if add_hm_colors:
        plot = plot * hm_segments_x
    if add_net_colors:
        plot = plot * net_segments_y
    return plot