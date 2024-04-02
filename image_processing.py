import hyperspy.api as hs
import atomap.initial_position_finding as ipf
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import atomap.api as am
import tkinter as tk
from tkinter.filedialog import askopenfilenames
import os, time
from skimage.restoration import estimate_sigma
from sklearn import decomposition
from scipy import fft, signal
from skimage.filters import butterworth
from skimage.restoration import denoise_nl_means # NL means
from skimage.restoration import richardson_lucy # RL deconvolution
from scipy.stats import binned_statistic
from scipy.signal import find_peaks
from matplotlib.widgets import SpanSelector
from matplotlib import gridspec
from hyperspy.component import Component
from sklearn.metrics import r2_score
from matplotlib.offsetbox import AnchoredText


hs.preferences.GUIs.warn_if_guis_are_missing = False
hs.preferences.save()
plt.rcParams['figure.figsize'] = (8,8)

def open_file():
    root = tk.Tk()
    root.attributes('-topmost',True)
    root.iconify()   
    file_path = askopenfilenames(parent=root)
    root.destroy()
    return file_path

def scaling(s, det_image):
    # Axes Scaling
    s.axes_manager[0].name = 'X'
    s.axes_manager[1].name = 'Y'
    s.axes_manager[0].scale = pixel_size_pm/1000
    s.axes_manager[1].scale = pixel_size_pm/1000
    s.axes_manager[0].units = 'nm'
    s.axes_manager[1].units = 'nm'
    s.metadata.General.title = ''

    # Intensity scaling
    s_normalised = am.quant.detector_normalisation(s, det_image, inner_angle=inner_angle, outer_angle=outer_angle)
    #s_normalised = s
    # Plot
    s_normalised.plot(colorbar=False)
    ax=plt.gca()
    norm = mpl.colors.Normalize(vmin=np.min(s_normalised.data), vmax=np.max(s_normalised.data))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.15)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='Greys_r'),ax=ax, pad=.05, fraction=.1,cax=cax)
    plt.tight_layout()

    #plt.savefig(path+'\\normalized_image.png',dpi=512,transparent=True,bbox_inches='tight')
    return s_normalised

def PCA(original_imag, n_components):
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(original_imag)
    pcaFaces = pca.transform(original_imag)
    pca_imag = pca.inverse_transform(pcaFaces)
    sigma_est = np.mean(estimate_sigma(pca_imag[pixels_cropped:-pixels_cropped,pixels_cropped:-pixels_cropped]))
    print(sigma_est)
    return pca_imag

def NL(original_imag, sigma_est):
    h=5
    patch_size=5
    patch_distance=6
    nlm_imag = denoise_nl_means(original_imag, h=h*sigma_est, fast_mode=True,patch_size=patch_size,patch_distance=patch_distance)
    return nlm_imag

def RL(original_imag, probe_resolution, iters):
    pixel_size=pixel_size_pm/1000
    width_pix=round(probe_resolution/pixel_size)
    kernlen = 4*width_pix
    std=width_pix/2.348
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    gaussian_probe= gkern2d
    #plt.figure(figsize=(4,4))
    #plt.imshow(gaussian_probe,cmap='gray')
    rl_imag = richardson_lucy(original_imag, gaussian_probe, iters)
    return rl_imag

def band_pass(image):
    image = image[pixels_cropped:-pixels_cropped,pixels_cropped:-pixels_cropped]
    high_pass=np.mean(np.abs(fft.ifft2((fft.fft2(image)==fft.fft2(image)[0,0])*fft.fft2(image))))+butterworth(image,0.005,True,order=2)
    band_pass=butterworth(high_pass,0.08,False,order=4)
    return band_pass
    
def compare(original_imag, filter_imag, text):

    sigma_est_original = np.mean(estimate_sigma(original_imag[pixels_cropped:-pixels_cropped,pixels_cropped:-pixels_cropped]))
    print(f'estimated noise standard deviation from original image = {sigma_est_original}')
    
    sigma_est = np.mean(estimate_sigma(filter_imag[pixels_cropped:-pixels_cropped,pixels_cropped:-pixels_cropped]))
    print(f'estimated noise standard deviation from {text} denoising = {sigma_est}')
    
    fig, ax = plt.subplots(figsize=(12,8), nrows=1, ncols=3)
    
    ax[0].set_title('Original Image')
    ax[0].imshow(original_imag[pixels_cropped:-pixels_cropped,pixels_cropped:-pixels_cropped], cmap='gray')
    ax[0].axis('off')
    
    
    ax[1].set_title(text)
    ax[1].imshow(filter_imag[pixels_cropped:-pixels_cropped,pixels_cropped:-pixels_cropped], cmap='gray')
    ax[1].axis('off')
    
    ax[2].set_title('Residuals')
    ax[2].imshow(original_imag[pixels_cropped:-pixels_cropped,pixels_cropped:-pixels_cropped]-filter_imag[pixels_cropped:-pixels_cropped,pixels_cropped:-pixels_cropped], cmap='gray')
    ax[2].axis('off')
    
    fig.tight_layout()
    plt.show()
    

def get_sublattice(s_normalised, optimal_separation, optimal_separation_d):
    atom_positions = am.get_atom_positions(s_normalised, optimal_separation,pca=True,subtract_background=True, normalize_intensity=True)
    sublattice = am.Sublattice(atom_positions, s_normalised)
    sublattice.find_nearest_neighbors()
    sublattice.refine_atom_positions_using_center_of_mass()
    #sublattice.refine_atom_positions_using_2d_gaussian()
    #sublattice.get_atom_list_on_image(markersize=5).plot()
    
    
    if dumbell is True:
        dumbbell_vector = ipf.find_dumbbell_vector(atom_positions)
        dumbbell_positions = am.get_atom_positions(s_normalised, optimal_separation_d,pca=True,subtract_background=True, normalize_intensity=True)
        sublattice = am.Sublattice(dumbbell_positions, s_normalised)
        #sublattice.get_atom_list_on_image(markersize=5).plot()

        # Dumbell recognition
        dumbbell_positions = np.asarray(dumbbell_positions)
        dumbbell_lattice = ipf.make_atom_lattice_dumbbell_structure(s_normalised, dumbbell_positions, dumbbell_vector)
        dumbbell_lattice.pixel_size=s_normalised.axes_manager[0].scale
        dumbbell_lattice.sublattice_list[0].pixel_size=s_normalised.axes_manager[0].scale
        dumbbell_lattice.sublattice_list[1].pixel_size=s_normalised.axes_manager[0].scale
        dumbbell_lattice.units=s_normalised.axes_manager[0].units
        dumbbell_lattice.sublattice_list[0].units=s_normalised.axes_manager[0].units
        dumbbell_lattice.sublattice_list[1].units=s_normalised.axes_manager[0].units
    else:
        return sublattice


    return dumbbell_lattice

def intensity_map(image, atom_lattice):
    sublattice_A=atom_lattice.sublattice_list[1]
    sublattice_B=atom_lattice.sublattice_list[0]
    atom_lattice.units=atom_lattice.sublattice_list[0].units
    atom_lattice.pixel_size=atom_lattice.sublattice_list[0].pixel_size
    
    # Intensity map of A 
    sublattice_A.original_image= image
    sublattice_A.find_nearest_neighbors(nearest_neighbors=15)
    sublattice_A._pixel_separation = sublattice_A._get_pixel_separation()
    sublattice_A._make_translation_symmetry()
    if ((0,0) in sublattice_A.zones_axis_average_distances):
        index=sublattice_A.zones_axis_average_distances.index((0,0))
        sublattice_A.zones_axis_average_distances.remove(sublattice_A.zones_axis_average_distances[index])
        sublattice_A.zones_axis_average_distances_names.remove(sublattice_A.zones_axis_average_distances_names[index])
    sublattice_A._generate_all_atom_plane_list(atom_plane_tolerance=0.5)
    sublattice_A._sort_atom_planes_by_zone_vector()
    sublattice_A._remove_bad_zone_vectors()
    
    
    direction=2

    zone_vector = sublattice_A.zones_axis_average_distances[direction]
    atom_planes = sublattice_A.atom_planes_by_zone_vector[zone_vector]
    zone_axis = sublattice_A.get_atom_planes_on_image(atom_planes)
    
    # Plot directions
    zone_axis.plot()
    ax = plt.gca()
    fig=plt.gcf()
    fig.set_size_inches((10,10))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()
    #######

    sublattice_A.find_sublattice_intensity_from_masked_image(sublattice_A.original_image,radius=7)
    zone_axis_A = sublattice_A.zones_axis_average_distances[direction]
    atom_plane_list_A = sublattice_A.atom_planes_by_zone_vector[zone_axis_A]
    intensity_A=[]
    x_values=[]
    y_values=[]
    for i in range(0,len(atom_plane_list_A)):
        atomplane=atom_plane_list_A[i]
        plane_intensity=[]
        x_values_plane=[]
        y_values_plane=[]
        for j in range(0, len(atomplane.atom_list)):
            atom=atomplane.atom_list[j]
            x_pos,y_pos=atom.get_pixel_position()
            intensity=atom.intensity_mask
            plane_intensity.append(intensity)
            x_values_plane.append(x_pos)
            y_values_plane.append(y_pos)
        intensity_A.append(plane_intensity)
        x_values.append(x_values_plane)
        y_values.append(y_values_plane)
    
    intensity_A_array = np.zeros([len(intensity_A),len(max(intensity_A,key = lambda x: len(x)))])
    intensity_A_array[:] = np.nan
    for i,j in enumerate(intensity_A):
        intensity_A_array[i][0:len(j)] = j
    
    x_values_array = np.zeros([len(x_values),len(max(x_values,key = lambda x: len(x)))])
    x_values_array[:] = np.nan
    for i,j in enumerate(x_values):
        x_values_array[i][0:len(j)] = j
    
    y_values_array = np.zeros([len(y_values),len(max(y_values,key = lambda x: len(x)))])
    y_values_array[:] = np.nan
    for i,j in enumerate(y_values):
        y_values_array[i][0:len(j)] = j
        
    intensity_A=np.stack((intensity_A_array,x_values_array,y_values_array),axis=2)

    # Intnsity of B sublattice
    
    sublattice_B.original_image=image
    sublattice_B.find_nearest_neighbors(nearest_neighbors=15)
    sublattice_B._pixel_separation = sublattice_B._get_pixel_separation()
    sublattice_B._make_translation_symmetry()
    if ((0,0) in sublattice_B.zones_axis_average_distances):
        index=sublattice_B.zones_axis_average_distances.index((0,0))
        sublattice_B.zones_axis_average_distances.remove(sublattice_B.zones_axis_average_distances[index])
        sublattice_B.zones_axis_average_distances_names.remove(sublattice_B.zones_axis_average_distances_names[index])
    sublattice_B._generate_all_atom_plane_list(atom_plane_tolerance=0.5)
    sublattice_B._sort_atom_planes_by_zone_vector()
    sublattice_B._remove_bad_zone_vectors()
    
    direction=2

    zone_vector = sublattice_B.zones_axis_average_distances[direction]
    atom_planes = sublattice_B.atom_planes_by_zone_vector[zone_vector]
    zone_axis = sublattice_B.get_atom_planes_on_image(atom_planes)
    

    ##  Plot axis
    zone_axis.plot()
    ax = plt.gca()
    fig=plt.gcf()
    fig.set_size_inches((10,10))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()
    ######


    
    sublattice_B.find_sublattice_intensity_from_masked_image(sublattice_B.original_image,radius=5)
    zone_axis_B = sublattice_B.zones_axis_average_distances[direction]
    atom_plane_list_B = sublattice_B.atom_planes_by_zone_vector[zone_axis_B]
    intensity_B=[]
    x_values=[]
    y_values=[]
    for i in range(0,len(atom_plane_list_B)):
        atomplane=atom_plane_list_B[i]
        plane_intensity=[]
        x_values_plane=[]
        y_values_plane=[]
        for j in range(0, len(atomplane.atom_list)):
            atom=atomplane.atom_list[j]
            x_pos,y_pos=atom.get_pixel_position()
            intensity=atom.intensity_mask
            plane_intensity.append(intensity)
            x_values_plane.append(x_pos)
            y_values_plane.append(y_pos)
        intensity_B.append(plane_intensity)
        x_values.append(x_values_plane)
        y_values.append(y_values_plane)
    
    intensity_B_array = np.zeros([len(intensity_B),len(max(intensity_B,key = lambda x: len(x)))])
    intensity_B_array[:] = np.nan
    for i,j in enumerate(intensity_B):
        intensity_B_array[i][0:len(j)] = j
    
    x_values_array = np.zeros([len(x_values),len(max(x_values,key = lambda x: len(x)))])
    x_values_array[:] = np.nan
    for i,j in enumerate(x_values):
        x_values_array[i][0:len(j)] = j
    
    y_values_array = np.zeros([len(y_values),len(max(y_values,key = lambda x: len(x)))])
    y_values_array[:] = np.nan
    for i,j in enumerate(y_values):
        y_values_array[i][0:len(j)] = j
        
    intensity_B=np.stack((intensity_B_array,x_values_array,y_values_array),axis=2)
    
    return intensity_A, intensity_B


def intensity_map2(image, atom_lattice):
    sublattice_B=atom_lattice
    atom_lattice.units=atom_lattice.sublattice_list[0].units
    atom_lattice.pixel_size=atom_lattice.sublattice_list[0].pixel_size
    


    # Intnsity of B sublattice
    
    sublattice_B.original_image=image
    sublattice_B.find_nearest_neighbors(nearest_neighbors=15)
    sublattice_B._pixel_separation = sublattice_B._get_pixel_separation()
    sublattice_B._make_translation_symmetry()
    if ((0,0) in sublattice_B.zones_axis_average_distances):
        index=sublattice_B.zones_axis_average_distances.index((0,0))
        sublattice_B.zones_axis_average_distances.remove(sublattice_B.zones_axis_average_distances[index])
        sublattice_B.zones_axis_average_distances_names.remove(sublattice_B.zones_axis_average_distances_names[index])
    sublattice_B._generate_all_atom_plane_list(atom_plane_tolerance=0.5)
    sublattice_B._sort_atom_planes_by_zone_vector()
    sublattice_B._remove_bad_zone_vectors()
    
    direction=2

    zone_vector = sublattice_B.zones_axis_average_distances[direction]
    atom_planes = sublattice_B.atom_planes_by_zone_vector[zone_vector]
    zone_axis = sublattice_B.get_atom_planes_on_image(atom_planes)
    

    ##  Plot axis
    zone_axis.plot()
    ax = plt.gca()
    fig=plt.gcf()
    fig.set_size_inches((10,10))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()
    ######


    
    sublattice_B.find_sublattice_intensity_from_masked_image(sublattice_B.original_image,radius=5)
    zone_axis_B = sublattice_B.zones_axis_average_distances[direction]
    atom_plane_list_B = sublattice_B.atom_planes_by_zone_vector[zone_axis_B]
    intensity_B=[]
    x_values=[]
    y_values=[]
    for i in range(0,len(atom_plane_list_B)):
        atomplane=atom_plane_list_B[i]
        plane_intensity=[]
        x_values_plane=[]
        y_values_plane=[]
        for j in range(0, len(atomplane.atom_list)):
            atom=atomplane.atom_list[j]
            x_pos,y_pos=atom.get_pixel_position()
            intensity=atom.intensity_mask
            plane_intensity.append(intensity)
            x_values_plane.append(x_pos)
            y_values_plane.append(y_pos)
        intensity_B.append(plane_intensity)
        x_values.append(x_values_plane)
        y_values.append(y_values_plane)
    
    intensity_B_array = np.zeros([len(intensity_B),len(max(intensity_B,key = lambda x: len(x)))])
    intensity_B_array[:] = np.nan
    for i,j in enumerate(intensity_B):
        intensity_B_array[i][0:len(j)] = j
    
    x_values_array = np.zeros([len(x_values),len(max(x_values,key = lambda x: len(x)))])
    x_values_array[:] = np.nan
    for i,j in enumerate(x_values):
        x_values_array[i][0:len(j)] = j
    
    y_values_array = np.zeros([len(y_values),len(max(y_values,key = lambda x: len(x)))])
    y_values_array[:] = np.nan
    for i,j in enumerate(y_values):
        y_values_array[i][0:len(j)] = j
        
    intensity_B=np.stack((intensity_B_array,x_values_array,y_values_array),axis=2)
    
    return intensity_B



# Muraki model stuff
def onselect(xmin, xmax):
    global x_pos 
    x_pos = np.array([xmin,xmax])
    
    
def mean_values(avg_intensity):
    count_binned=binned_statistic(avg_intensity,avg_intensity, 'count', bins=10)
    bin_centers=(count_binned[1][1:] + count_binned[1][:-1])/2
    mean_binned=binned_statistic(avg_intensity,avg_intensity, 'mean', bins=10)
    pos_peaks, _ = find_peaks(count_binned[0], height=0)
    pos_peaks_sorted=pos_peaks[np.argsort(count_binned[0][pos_peaks])]
    peaks_sorted=mean_binned[0][pos_peaks_sorted]
    
    n_lower_limit=3
    n_upper_limit=-2
    
    lower_limit,upper_limit=count_binned[1][n_lower_limit],count_binned[1][n_upper_limit]
    positions_l=np.where(avg_intensity<lower_limit)
    i_barriers=np.nanmean(avg_intensity[positions_l])
    positions_u=np.where(avg_intensity>upper_limit)
    i_quantum_well=np.nanmean(avg_intensity[positions_u])
    print('Mean intensity of the barriers: '+str(i_barriers))
    print('Mean intensity of the quantum well: '+str(i_quantum_well))
    return i_barriers, i_quantum_well

class Muraki(Component):
    def __init__(self, parameter_1=1, parameter_2=2, parameter_3=3):
        Component.__init__(self, ('x0', 's', 'N'))
        self.x0.value = 1
        self.s.value = 0.5
        self.N.value = 5
        self.x0.bmin = 0
        self.x0.bmax = 1
        self.s.bmin = 0
        self.s.bmax = 1
        self.N.bmin = 0
        self.N.bmax = 50
    def function(self, x):
        x0 = self.x0.value
        s = self.s.value
        N = self.N.value
        return np.piecewise(x,[((x >= 1.0) & (x<= N)),x >= N],[lambda x : x0*(1.0 -s**x), lambda x: x0*(1 -s**x)*s**(x-N)])

def f(x,x0,s,N):
    return np.piecewise(x,[((x >= 1.0) & (x<= N)),x >= N],[lambda x : x0*(1.0 -s**x), lambda x: x0*(1 -s**x)*s**(x-N)])

def composition_profile(intensity_A, intensity_B, atom_lattice):
    global avg_axis, avg_intensity, avg_std, avg_axis1, avg_intensity1, avg_std1
    # Parameters
    avg_intensity=np.nanmean(intensity_A[:,:,0],axis=1)
    avg_std=np.nanstd(intensity_A[:,:,0],axis=1)
    avg_axis=np.nanmean(intensity_A[:,:,2],axis=1)*atom_lattice.pixel_size
    avg_intensity1=np.nanmean(intensity_B[:,:,0],axis=1)
    avg_std1=np.nanstd(intensity_B[:,:,0],axis=1)
    avg_axis1=np.nanmean(intensity_B[:,:,2],axis=1)*atom_lattice.pixel_size
    

    #
    intensity_map = intensity_A
    nominal_composition=1
    i_barriers, i_quantum_well = mean_values(avg_intensity)
    
    
    normalized_array=(intensity_map-i_barriers)/(i_quantum_well-i_barriers)
    avg_norm=np.nanmean(normalized_array[:,:,0],axis=1)
    std_dev_norm=np.nanstd(normalized_array[:,:,0],axis=1)
    
    
    # Selection of the QW
    fig = plt.figure(figsize=(14, 8)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1]) 
    ax0 = plt.subplot(gs[0])
    img1=ax0.plot(avg_norm,'*--')
    ax0.set(xlabel='Layer',ylabel='Average Composition')

    span = SpanSelector(
    ax0,
    onselect,
    "horizontal",
    useblit=True,
    props=dict(alpha=0.5, facecolor="tab:blue"),
    interactive=True,
    drag_from_anywhere=True
    )
    
    ax1 = plt.subplot(gs[1])
    img2=ax1.scatter(intensity_map[:,:,1],intensity_map[:,:,2],s=20,c=normalized_array[:,:,0],cmap='jet',vmin=-0.25,vmax=1.25)
    fig.colorbar(img2,shrink=0.4,pad=0)
    ax1.axis('scaled')
    ax1.axis('off')
    ax1.set_ylim(ax1.get_ylim()[::-1]) 
    plt.tight_layout()
    plt.show(block=False)
    
    
    
    # Signal to fit with Muraki model
    muraki_positions=np.arange(x_pos[0]+1,x_pos[1]+1,dtype=int)
    muraki_signal=avg_norm[muraki_positions]
    std_dev=std_dev_norm[muraki_positions]
    sc=hs.signals.Signal1D(muraki_signal)
    print('Lower layer of the selection: '+str(muraki_positions[0]))
    print('Upper layer of the selection: '+str(muraki_positions[-1]))

    # 
    muraki_model = sc.create_model()
    muraki = Muraki()
    muraki_model.append(muraki)
    muraki_model.fit()
    muraki_model.print_current_values()

    x=np.arange(0,len(sc.data),dtype=float)
    y_pred=f(x,muraki.x0.value,muraki.s.value,muraki.N.value)
    r2_parameter=r2_score(sc.data[0::], y_pred[0::])    

    plt.figure()
    plt.plot(avg_axis[np.arange(0,len(avg_intensity))],avg_norm,'*--')
    plt.plot(avg_axis[x.astype(int)+muraki_positions[0]],y_pred,'-',color='red')
    plt.xlabel('Position [nm]')
    plt.ylabel('Average Composition')
    plt.minorticks_on()
    plot=plt.gca()
    label='$R^2 = $'+str(np.round(r2_parameter,3))
    at = AnchoredText(label, prop=dict(size=10), frameon=True, loc='upper right')
    at.patch.set(edgecolor='lightgray')
    at.patch.set_boxstyle('round,pad=0.,rounding_size=0.1')
    plot.add_artist(at)
    plt.show(block=False)
    

def plot_profile(intensity_A,intensity_B):

    # Create figure and axes with custom size
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [2.5, 2.5, 4]})
    
    # First subplot
    scatter1 = axs[0].scatter(intensity_A[:,:,1], intensity_A[:,:,2], s=20, c=intensity_A[:,:,0], cmap='gnuplot')
    #plt.colorbar(scatter1, ax=axs[0], shrink=0.5, pad=-0.18)
    axs[0].set_aspect('equal')
    axs[0].axis('off')
    axs[0].xaxis.tick_top()
    axs[0].yaxis.tick_left()
    axs[0].set_ylim(axs[0].get_ylim()[::-1]) 
    axs[0].set_title('Group III')
    
    # Second subplot
    scatter2 = axs[1].scatter(intensity_B[:,:,1], intensity_B[:,:,2], s=20, c=intensity_B[:,:,0], cmap='jet')
    #plt.colorbar(scatter2, ax=axs[1], shrink=0.5, pad=-0.18)
    axs[1].set_aspect('equal')
    axs[1].axis('off')
    axs[1].xaxis.tick_top() 
    axs[1].yaxis.tick_left()
    axs[1].set_ylim(axs[1].get_ylim()[::-1]) 
    axs[1].set_title('Group V')
    
    # Third subplot
    axs[2].errorbar(avg_axis, avg_intensity, yerr=avg_std, ecolor='lightcoral', marker='.', fmt=':', capsize=3, alpha=0.75, mec='red', mfc='red', color='red')
    axs[2].fill_between(avg_axis, avg_intensity - avg_std, avg_intensity + avg_std, alpha=.25, color='lightcoral')
    axs[2].errorbar(avg_axis1, avg_intensity1, yerr=avg_std1, ecolor='lightblue', marker='.', fmt=':', capsize=3, alpha=0.75, mec='skyblue', mfc='skyblue', color='skyblue')
    axs[2].fill_between(avg_axis1, avg_intensity1 - avg_std1, avg_intensity1 + avg_std1, alpha=.25, color='lightblue')
    axs[2].set_xlabel('Position [nm]')
    axs[2].set_ylabel('Intensity')
    axs[2].minorticks_on()
    axs[2].grid(which='both', linestyle='--', color='gray', alpha=0.5)
    axs[2].set_title('Intensity vs Position')
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the figure
    plt.show(block=False)




if __name__ == '__main__':

    global pixels_cropped, pixel_size_pm, inner_angle, outer_angle
    pixel_size_pm=6.652 #3.326 #6.652 # pm
    inner_angle=130
    outer_angle=200
    pixels_cropped=50

    file_paths= open_file()
    det_path= open_file()
    det_image=hs.signals.Signal2D(plt.imread(det_path[0]))

    global dumbell
    dumbell = False

    for file_path in file_paths:
    
        s=hs.load(file_path)
        s = s.isig[10:1000, 10:3000]
        
        path = os.path.splitext(file_path)[0]
        if not (os.path.exists(path)):
            os.mkdir(path)

        
        s_normalised = scaling(s, det_image)
        original_imag=s_normalised.data
        original_imag=original_imag[pixels_cropped:-pixels_cropped,pixels_cropped:-pixels_cropped]
        #sigma_est = np.mean(estimate_sigma(original_imag))
        
        
        pca_imag = PCA(original_imag, n_components = 16)
        #rl_imag = RL(original_imag, probe_resolution = 0.065, iters = 5)
        #nl_imag =  NL(original_imag, sigma_est)
        
        band_pass_pca_imag = band_pass(pca_imag)
        
        # Find sublattice
        #st = time.time()
        s_normalised.data=original_imag
        if (os.path.exists(path+'\\data.hdf5')):
            atom_lattice = am.load_atom_lattice_from_hdf5(path+'\\data.hdf5',construct_zone_axes=False)
        else:
            atom_lattice = get_sublattice(s_normalised, optimal_separation = 18, optimal_separation_d = 24)
            atom_lattice.save(path+'\\data.hdf5', overwrite=True)

        # Intensity map
        images = ["pca_imag"]
        for image_name in images:

            if (os.path.exists(path+'\\im_A_'+image_name+'.npy')) & (os.path.exists(path+'\\im_B_'+image_name+'.npy')):
                intensity_A = np.load(path+'\\im_A_'+image_name+'.npy')
                intensity_B = np.load(path+'\\im_B_'+image_name+'.npy')
            else:
                intensity_A, intensity_B = intensity_map(globals()[image_name], atom_lattice)
                np.save(path+'\\im_A_'+image_name+'.npy',intensity_A)
                np.save(path+'\\im_B_'+image_name+'.npy',intensity_B)
            compare(original_imag , pca_imag, 'PCA - 8')
            intensity_A, intensity_B = intensity_B, intensity_A
            composition_profile(intensity_A, intensity_B, atom_lattice)
            #et = time.time()
            #print(elapsed_time = et - st)
        
        
        #plot_profile(intensity_A,intensity_B)
        #print('end')
    
    
    
    
    
    
    
    
    
    