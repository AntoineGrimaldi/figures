import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter, writers
from tqdm.notebook import tqdm
from moviepy.editor import VideoFileClip, ImageClip, ImageSequenceClip, TextClip, CompositeVideoClip, ColorClip

def norm(X):
    return (X-X.min())/(X.max()-X.min())

def vonmises(N_inputs, A, theta, k=2):
    return A*norm(np.exp(k*np.cos(2*np.pi*(np.linspace(0, 1, N_inputs)-theta))))

def cospattern(N_inputs, A, theta, k = 4):
    return A*np.cos(np.linspace(0,k*np.pi,N_inputs)+theta)

def linear(N_inputs, A, theta):
    return np.linspace(0,A,N_inputs)

def make_input(nb_syn, noise_density, nb_stim, simtime, patwindow, function=cospattern):
    # draw random gaussian noise spike timings -> shape (nb_syn, nb_ev_noise)
    noise = (np.random.random((nb_syn, int(noise_density*simtime)))*simtime).astype(int)
    # make time references for stimulus repetition -> pattime
    repet = np.linspace(1,nb_stim+1, nb_stim)*simtime/(nb_stim + 2)
    pattime = np.tile(repet,(nb_syn,1))
    # draw stimulus -> stim
    theta = 0
    pattern = np.tile(function(nb_syn, patwindow, theta), (nb_stim,1)).T
    stim = (pattime + pattern).astype(int)
    # make address event representation
    time = (np.hstack([noise,stim]).ravel())
    addr = np.repeat(np.arange(0,nb_syn),stim.shape[1]+noise.shape[1])
    aer = (addr[np.argsort(time)],time[np.argsort(time)])
    return noise, stim, aer

def membrane_potential(aer, simtime, delays, tau, weight):
    addresses, timestamps = aer
    delayed_timestamps = timestamps + delays[addresses]

    sorted_times = np.sort(delayed_timestamps)
    dts = np.diff(np.hstack((0, sorted_times)))
    dts = np.diff(np.arange(simtime))
    V = np.zeros_like(dts)
    for i, dt in enumerate(dts):
        spike_indice = np.where(sorted_times==i)[0]
        #print(spike_indice.size, spike_indice)
        if i==0: 
            V[i] = 0
        else:
            if V[i-1]>1: 
                V[i] = 0.
            elif spike_indice.size>0:
                V[i] = np.exp( - dt / tau) * V[i-1] + weight[addresses[spike_indice[0]]]
                if spike_indice.size>1:
                    for nb_spike in range(1,len(spike_indice)):
                        V[i] += weight[addresses[spike_indice[nb_spike]]]
            else:
                V[i] = np.exp( - dt / tau) * V[i-1]
    return sorted_times, V
    
def printfigLIF(fig, name, width=1500, height=1000, dpi_exp=100, bbox='tight', path='LIF_figures/'):
    #path = '../../GrimaldiEtAl2020HOTS_clone_laurent/fig'
    #fig.set_size_inches(width/dpi_exp, height/dpi_exp)
    fig.savefig(path+name, dpi=dpi_exp)#, bbox_inches=bbox, transparent=True)
    
def printfigHSD(fig, name, width=1500, height=1000, dpi_exp=100, bbox='tight', path='HSD_figures/'):
    #path = '../../GrimaldiEtAl2020HOTS_clone_laurent/fig'
    #fig.set_size_inches(width/dpi_exp, height/dpi_exp)
    fig.savefig(path+name, dpi=dpi_exp)#, bbox_inches=bbox, transparent=True)
        
def draw_LIF_figures(noise, stim, aer, simtime, nb_syn, delays, synaptic_weights, nb_frames=200, width=1500, height=1000, dpi_exp=100, print_figures=True, write_movie=True, path = 'LIF_figures/'):
    colorz = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'lightcoral']
    color_spike_pre = 'black'
    color_spike_post = colorz[1]
    color_spike_out = colorz[3]
    color_synapse = 'black'
    color_neuron = 'black'
    color_stim = colorz[2]
    color_V = colorz[1]
    alpha_V_futur = .3
    alpha_spike_post = .5
    tau = 300
    seed_nb = 0
    np.random.seed(seed_nb)
    fig_height, fig_width =  height/dpi_exp, width/dpi_exp
    fps = 30
    dpi_exp = 100

    linewidth = 1
    bottom = .1
    top = .9
    raster_plot_time = .5
    synapses_max_delays = .25
    output_plot_time = .25
    neuron_width = .1
    h_spacing = (top-bottom)/nb_syn
    spik_height = .8*h_spacing
    max_synapse_weight = .05
    presyn_window_size = 1000
    postsyn_window_size = 300
    postsoma_window_size = 300
    post_spik_width = max_synapse_weight
    ratio = fig_height/(fig_width*2)
    
    frame_interval = simtime/nb_frames
    
    noise = simtime-noise
    if stim.size>0:
        stim = simtime-stim
    sorted_times, V = membrane_potential(aer, simtime, delays, tau, synaptic_weights)
    shifted_V = np.hstack((np.zeros([2*(postsyn_window_size+presyn_window_size)-postsoma_window_size]),V))
    output_spikes = np.where(shifted_V>=1)[0]
    
    time = 0
    if write_movie:
        sequence = []
    for frame in tqdm(range(nb_frames)):
        
        if print_figures:
            fig, ax = plt.subplots(2,1, figsize=(fig_width, fig_height))
            ax[0].axis('off')
            output_line = plt.Line2D([raster_plot_time+synapses_max_delays, 1],[.5, .5], linewidth=linewidth, c='k')
            ax[0].add_artist(output_line)
            time += frame_interval
            time_shift = simtime-time
            shifted_noise = noise-time_shift
            if stim.size>0:
                shifted_stim = stim-time_shift

            start_time_window = int(time)
            present_time = int(time+postsyn_window_size+presyn_window_size)
            stop_time_window = int(time+postsoma_window_size+postsyn_window_size+presyn_window_size)

            output_plots = output_spikes[(output_spikes>present_time-postsoma_window_size) & (output_spikes<present_time)]
            if output_plots.size>0:
                for out_spik in range(len(output_plots)):
                    position = raster_plot_time+synapses_max_delays+output_plot_time*(present_time-output_plots[out_spik])/postsoma_window_size
                    output_spik = plt.Line2D([position,position],[.5, .5+spik_height], linewidth=linewidth, c=color_spike_out)
                    ax[0].add_artist(output_spik)
            neuron = Ellipse((raster_plot_time+synapses_max_delays, .5), neuron_width, neuron_width/ratio, color='w', ec=color_neuron, zorder=4)
            ax[0].add_artist(neuron);
            for syn in range(nb_syn):
                presyn_lines = plt.Line2D([0, raster_plot_time],[bottom+syn*h_spacing, bottom+syn*h_spacing], linewidth=linewidth, c=color_synapse)
                ax[0].add_artist(presyn_lines)
                synapses_lines = plt.Line2D([raster_plot_time, raster_plot_time+synapses_max_delays],[bottom+syn*h_spacing, .5], linewidth=linewidth, c=color_synapse)
                ax[0].add_artist(synapses_lines)
                synapses = Ellipse((raster_plot_time, bottom+syn*h_spacing), max_synapse_weight*synaptic_weights[syn], max_synapse_weight*synaptic_weights[syn]/ratio, color=color_synapse, ec=color_synapse, zorder=4)
                ax[0].add_artist(synapses)
                for noise_spik in range(len(shifted_noise[syn])):
                    if shifted_noise[syn][noise_spik]>0 and shifted_noise[syn][noise_spik]<presyn_window_size:
                        presyn_spikes = plt.Line2D([raster_plot_time*shifted_noise[syn][noise_spik]/presyn_window_size, raster_plot_time*shifted_noise[syn][noise_spik]/presyn_window_size],[bottom+syn*h_spacing, bottom+syn*h_spacing+spik_height], linewidth=linewidth, c=color_spike_pre)
                        ax[0].add_artist(presyn_spikes)
                    elif shifted_noise[syn][noise_spik]>=presyn_window_size and shifted_noise[syn][noise_spik]<presyn_window_size+postsyn_window_size:
                        x_position = raster_plot_time + synapses_max_delays*(shifted_noise[syn][noise_spik]-presyn_window_size)/postsyn_window_size
                        a = (.5-(bottom+syn*h_spacing))/synapses_max_delays
                        y_position = a*x_position + .5 - a*(synapses_max_delays+raster_plot_time)
                        postsyn_spikes = Ellipse((x_position, y_position), post_spik_width*synaptic_weights[syn], post_spik_width*synaptic_weights[syn]/ratio, color=color_spike_post, ec=color_spike_post, alpha = alpha_spike_post, zorder=4)
                        ax[0].add_artist(postsyn_spikes)
                if stim.size>0:
                    for stim_spik in range(len(shifted_stim[syn])):
                        if shifted_stim[syn][stim_spik]>0 and shifted_stim[syn][stim_spik]<presyn_window_size:
                            presyn_spikes_stim = plt.Line2D([raster_plot_time*shifted_stim[syn][stim_spik]/presyn_window_size, raster_plot_time*shifted_stim[syn][stim_spik]/presyn_window_size],[bottom+syn*h_spacing, bottom+syn*h_spacing+spik_height], linewidth=linewidth, c=color_stim)
                            ax[0].add_artist(presyn_spikes_stim)
                        elif shifted_stim[syn][stim_spik]>=presyn_window_size and shifted_stim[syn][stim_spik]<presyn_window_size+postsyn_window_size:
                            x_position = raster_plot_time + synapses_max_delays*(shifted_stim[syn][stim_spik]-presyn_window_size)/postsyn_window_size
                            a = (.5-(bottom+syn*h_spacing))/synapses_max_delays
                            y_position = a*x_position + .5 - a*(synapses_max_delays+raster_plot_time)
                            postsyn_spikes_stim = Ellipse((x_position, y_position), post_spik_width*synaptic_weights[syn], post_spik_width*synaptic_weights[syn]/ratio, color=color_stim, ec=color_stim, alpha = alpha_spike_post, zorder=4)
                            ax[0].add_artist(postsyn_spikes_stim)
            ymax = 1.3
            ax[1].plot(np.arange(simtime+2*(postsyn_window_size+presyn_window_size)-postsoma_window_size)[start_time_window:present_time], shifted_V[start_time_window:present_time], color=color_V)
            ax[1].plot(np.arange(simtime+2*(postsyn_window_size+presyn_window_size)-postsoma_window_size)[present_time:stop_time_window], shifted_V[present_time:stop_time_window], color=color_V, alpha=alpha_V_futur)

            if output_plots.size>0:
                for V_spik in range(len(output_plots)):
                    ax[1].vlines(output_plots[V_spik], 0, ymax, color = color_spike_out)
            ax[1].set_xlabel('time', fontsize=16)
            ax[1].set_ylabel('membrane potential', fontsize=16)
            ax[1].set_ylim(0,ymax)
            fig.suptitle('Leaky Integrate and Fire Neuron', fontsize=30)
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9)
            if print_figures:
                fig.savefig(path+f'LIF_{frame}.png', dpi=dpi_exp)
            if frame == int(nb_frames/2):
                plt.show()
            plt.close()
        if write_movie:
            sequence.append(path+f'LIF_{frame}.png')
    if write_movie:
        print('Making movie animation.mpg - this make take a while')
        os.system("mencoder 'LIF_figures/*.png' -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o animation.mpg")
        
def draw_HSD_box_figures(noise, stim, aer, simtime, nb_syn, delays, synaptic_weights, nb_frames=200, width=1500, height=1000, dpi_exp=100, print_figures=False):
    # esthetics
    colorz = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'lightcoral']
    color_spike_pre = 'black'
    color_spike_post = colorz[1]
    color_spike_out = colorz[3]
    color_synapse = 'black'
    color_neuron = 'black'
    color_stim = colorz[2]
    color_V = colorz[1]
    color_delay_box = colorz[0]
    alpha_V_futur = .3
    alpha_spike_post = .5
    fig_height, fig_width = 5*2, 15
    linewidth = 1
    spikewidth = 2
    delaywidth=16
    bottom = .1
    top = .9
    raster_plot_width = .5
    synapses_width = .25
    output_plot_width = .25
    neuron_width = .1
    h_spacing = (top-bottom)/nb_syn
    spik_height = .8*h_spacing
    max_synapse_weight = .05
    post_spik_width = max_synapse_weight
    ratio = fig_height/(fig_width*2)
    delay_box_vshift = .03
    delay_box_hshift = .01
    
    # parameters
    tau = 300
    seed_nb = 0
    np.random.seed(seed_nb)
    presyn_window_size = 1000
    postsyn_window_size = 300
    postsoma_window_size = 300
    frame_interval = simtime/nb_frames
    
    noise = simtime-noise
    if stim.size>0:
        stim = simtime-stim
    sorted_times, V = membrane_potential(aer, simtime, delays, tau, synaptic_weights)
    shifted_V = np.hstack((np.zeros([2*(int(3/2*postsyn_window_size)+presyn_window_size)-postsoma_window_size]),V))
    output_spikes = np.where(shifted_V>=1)[0]
    
    time = 0
    for frame in tqdm(range(nb_frames)):
        fig, ax = plt.subplots(2,1, figsize=(fig_width, fig_height))
        ax[0].axis('off')
        output_line = plt.Line2D([raster_plot_width+synapses_width, 1],[.5, .5], linewidth=linewidth, c='k')
        ax[0].add_artist(output_line)
        delay_box = Rectangle((raster_plot_width/2,0.05),raster_plot_width/2,.9, ec=color_delay_box, color='white',zorder=4)
        ax[0].add_artist(delay_box)
        time += frame_interval
        time_shift = simtime-time
        shifted_noise = noise-time_shift
        if stim.size>0:
            shifted_stim = stim-time_shift
        
        start_time_window = int(time)
        present_time = int(time+postsyn_window_size+presyn_window_size)
        stop_time_window = int(time+postsoma_window_size+postsyn_window_size+presyn_window_size)
        
        output_plots = output_spikes[(output_spikes>present_time-postsoma_window_size) & (output_spikes<present_time)]
        if output_plots.size>0:
            for out_spik in range(len(output_plots)):
                position = raster_plot_width+synapses_width+output_plot_width*(present_time-output_plots[out_spik])/postsoma_window_size
                output_spik = plt.Line2D([position,position],[.5, .5+spik_height], linewidth=spikewidth, c=color_spike_out)
                ax[0].add_artist(output_spik)
        neuron = Ellipse((raster_plot_width+synapses_width, .5), neuron_width, neuron_width/ratio, color='w', ec=color_neuron, zorder=4)
        ax[0].add_artist(neuron);
        for syn in range(nb_syn):
            delayed_stim = shifted_stim[syn] - int(delays[syn])
            delayed_noise = shifted_noise[syn] - int(delays[syn])
            presyn_lines = plt.Line2D([0, raster_plot_width],[bottom+syn*h_spacing, bottom+syn*h_spacing], linewidth=linewidth, c=color_synapse)
            ax[0].add_artist(presyn_lines)
            synapses_lines = plt.Line2D([raster_plot_width, raster_plot_width+synapses_width],[bottom+syn*h_spacing, .5], linewidth=linewidth, c=color_synapse)
            ax[0].add_artist(synapses_lines)
            synapses = Ellipse((raster_plot_width, bottom+syn*h_spacing), max_synapse_weight*synaptic_weights[syn], max_synapse_weight*synaptic_weights[syn]/ratio, color=color_synapse, ec=color_synapse, zorder=4)
            ax[0].add_artist(synapses)
            delay_stops = plt.Line2D([1.5*delay_box_hshift+raster_plot_width/2*(1+(presyn_window_size/2+delays[syn])/presyn_window_size), 1.5*delay_box_hshift+raster_plot_width/2*(1+(presyn_window_size/2+delays[syn])/presyn_window_size)],[bottom+syn*h_spacing, bottom+syn*h_spacing+spik_height], linewidth=spikewidth, c=color_delay_box, zorder=5)
            ax[0].add_artist(delay_stops)
            for noise_spik in range(len(shifted_noise[syn])):
                if shifted_noise[syn][noise_spik]>0 and shifted_noise[syn][noise_spik]<presyn_window_size/2:
                    presyn_spikes = plt.Line2D([raster_plot_width*shifted_noise[syn][noise_spik]/presyn_window_size, raster_plot_width*shifted_noise[syn][noise_spik]/presyn_window_size],[bottom+syn*h_spacing, bottom+syn*h_spacing+spik_height], linewidth=spikewidth, c=color_spike_pre)
                    ax[0].add_artist(presyn_spikes)
                elif delayed_noise[noise_spik]>=presyn_window_size and delayed_noise[noise_spik]<presyn_window_size+postsyn_window_size:
                    x_position = raster_plot_width + synapses_width*(delayed_noise[noise_spik]-presyn_window_size)/postsyn_window_size
                    a = (.5-(bottom+syn*h_spacing))/synapses_width
                    y_position = a*x_position + .5 - a*(synapses_width+raster_plot_width)
                    postsyn_spikes = Ellipse((x_position, y_position), post_spik_width*synaptic_weights[syn], post_spik_width*synaptic_weights[syn]/ratio, color=color_spike_post, ec=color_spike_post, alpha = alpha_spike_post, zorder=4)
                    ax[0].add_artist(postsyn_spikes)
                if shifted_noise[syn][noise_spik]>presyn_window_size/2 and shifted_noise[syn][noise_spik]<presyn_window_size+delays[syn]:
                    delay_lines = plt.Line2D([delay_box_hshift+raster_plot_width/2, delay_box_hshift+raster_plot_width/2*(1+(shifted_noise[syn][noise_spik]-presyn_window_size/2)/(presyn_window_size))], [delay_box_vshift+bottom+syn*h_spacing, delay_box_vshift+bottom+syn*h_spacing], linewidth=delaywidth, alpha=.3, c=color_delay_box, zorder=5)
                    ax[0].add_artist(delay_lines)
            if stim.size>0:
                for stim_spik in range(len(shifted_stim[syn])):
                    if shifted_stim[syn][stim_spik]>0 and shifted_stim[syn][stim_spik]<presyn_window_size/2:
                        presyn_spikes_stim = plt.Line2D([raster_plot_width*shifted_stim[syn][stim_spik]/presyn_window_size, raster_plot_width*shifted_stim[syn][stim_spik]/presyn_window_size],[bottom+syn*h_spacing, bottom+syn*h_spacing+spik_height], linewidth=spikewidth, c=color_stim)
                        ax[0].add_artist(presyn_spikes_stim)
                    elif delayed_stim[stim_spik]>=presyn_window_size and delayed_stim[stim_spik]<presyn_window_size+postsyn_window_size:
                        x_position = raster_plot_width + synapses_width*(delayed_stim[stim_spik]-presyn_window_size)/postsyn_window_size
                        a = (.5-(bottom+syn*h_spacing))/synapses_width
                        y_position = a*x_position + .5 - a*(synapses_width+raster_plot_width)
                        postsyn_spikes_stim = Ellipse((x_position, y_position), post_spik_width*synaptic_weights[syn], post_spik_width*synaptic_weights[syn]/ratio, color=color_stim, ec=color_stim, alpha = alpha_spike_post, zorder=4)
                        ax[0].add_artist(postsyn_spikes_stim)
                    if shifted_stim[syn][stim_spik]>presyn_window_size/2 and shifted_stim[syn][stim_spik]<presyn_window_size+delays[syn]:
                        delay_lines = plt.Line2D([delay_box_hshift+raster_plot_width/2, delay_box_hshift+raster_plot_width/2*(1+(shifted_stim[syn][stim_spik]-presyn_window_size/2)/(presyn_window_size))], [delay_box_vshift+bottom+syn*h_spacing, delay_box_vshift+bottom+syn*h_spacing], linewidth=delaywidth, alpha=.3, c=color_stim, zorder=5)
                        ax[0].add_artist(delay_lines)
        
        ymax = 1.3
        ax[1].plot(np.arange(simtime+2*(postsyn_window_size+presyn_window_size)-postsoma_window_size)[start_time_window:present_time], shifted_V[start_time_window:present_time], color=color_V)
        ax[1].plot(np.arange(simtime+2*(postsyn_window_size+presyn_window_size)-postsoma_window_size)[present_time:stop_time_window], shifted_V[present_time:stop_time_window], color=color_V, alpha=alpha_V_futur)
        
        if output_plots.size>0:
            for V_spik in range(len(output_plots)):
                ax[1].vlines(output_plots[V_spik], 0, ymax, color = color_spike_out)
        ax[1].set_xlabel('time', fontsize=16)
        ax[1].set_ylabel('membrane potential', fontsize=16)
        ax[1].set_ylim(0,ymax)
        fig.suptitle('Hetero-Synpatic Delays Neuron', fontsize=30)
        if print_figures:
            printfigHSD(fig,'HSD_box_'+str(frame)+'.png', width=width, height=height, dpi_exp=dpi_exp)
        else:
            if frame == int(nb_frames/2):
                plt.show()
        plt.close()
        
        
########################## COMPILE WITH SLIDES #################################        
################################################################################

class Slide:
    def __init__(self, contents=[], subtitles=[], fontsizes=[],
                 type='text', color='black', bg_color='white', duration=1):
        self.type = type
        self.subtitles = subtitles
        self.duration = duration
        self.contents = contents
        self.fontsizes = fontsizes
        self.color = color
        self.bg_color = bg_color

################################################################################
# http://zulko.github.io/moviepy/ref/VideoClip/VideoClip.html
################################################################################
class Deck:
    def __init__(self,
                 videoname = "no_name", fps=5, do_gif=False,
                 figpath = '',
                 H=500, W=800, border_ratio=.12):

        self.W = W
        self.H = H
        self.H_fig, self.W_fig = int(H-H*border_ratio), int(W-W*border_ratio)

        self.videoname = videoname
        self.do_gif = do_gif
        if self.do_gif: self.gifname = videoname + ".gif"
        self.figpath = figpath
        self.fps = fps


        self.txt_opts = dict(font="Open-Sans-Regular", align='center',
                             color='Black',
                             size=(W,H), method='caption')
        #self.txt_opts = dict(fontsize=65, bg_color='white', align='center', **opt_st)
        self.sub_opts = dict(font="Open-Sans-SemiBold", fontsize=30, align='South',
                                #bg_color='white',
                                color='orange',
                                size=(W,H), method='caption')

    def compositing(self, slides):
        t = 0
        clips = []
        for slide in slides:
            # contains figures or text
            if len(slide.contents)>0:
                # background
                clip = ColorClip(color=(255, 255, 255), size=(self.W, self.H))
                clip = clip.with_start(t).with_duration(slide.duration)
                clips.append(clip)

                sub_duration = slide.duration / len(slide.contents)
                for i_, content in enumerate(slide.contents):
                    if slide.type == 'text':
                        if len(slide.fontsizes)==0:
                            fontsize = 35 # default
                        elif len(slide.fontsizes)==1:
                            fontsize = slide.fontsizes
                        else:
                            fontsize = slide.fontsizes[i_]

                        clip = TextClip(content, bg_color=slide.bg_color,**self.txt_opts)
                                        #fontsize=fontsize, **self.txt_opts)
                    else:
                        # drawing the list of figures
                        clip = ImageClip(os.path.join(self.figpath, content))
                    # time
                    clip = clip.with_start(t).with_duration(sub_duration)
                    # space
                    clip = clip.resize(height=self.H_fig, width=self.W_fig)
                    #clip = clip.set_pos(("center", "top"))

                    clips.append(clip)
                    t += sub_duration

            if len(slide.subtitles)>0:
                # overlaying subtitles
                t -= slide.duration # be kind, rewind
                sub_duration = slide.duration / len(slide.subtitles)
                for subtitle in slide.subtitles:
                    sub = TextClip(subtitle, **self.sub_opts)
                    # time
                    sub = sub.with_start(t).with_duration(sub_duration)
                    # space
                    sub = sub.resize(height=self.H_fig, width=self.W_fig)
                    #sub = sub.set_pos('center')
                    
                    #sub = sub.on_color(size=(clip.w + sub.w, sub.h-10),
                    #                   color=(0,0,0), pos=(6,'center'), col_opacity=0.6)

                    clips.append(sub)
                    t += sub_duration
            else:
                print('/!\ no subtitle', slide.contents)

        return clips

    def compiling(self, clips):
        # Compostiting all clips
        video = CompositeVideoClip(clips)
        print('Writing', self.videoname + '.mp4')
        video.write_videofile(self.videoname + '.mp4', fps=self.fps)
        if self.do_gif:
            print('Writing', self.gifname)
            video.write_gif(self.gifname, fps=self.fps)
            from pygifsicle import optimize
            optimize(self.gifname)
            print('Done')