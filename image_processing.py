import hyperspy.api as hs
import atomap.api as am
import matplotlib.pyplot as plt
import os, time
import numpy as np

from setting_image import open_file, imp, compare
from atomap_image import get_sublattice, intensity_map, intensity_map2, plot_lattice
from profile_image import fitt_muraki, composition_profile, plot_profile

st = time.time()
hs.preferences.GUIs.warn_if_guis_are_missing = False
hs.preferences.save()
plt.rcParams['figure.figsize'] = (8,8)


def make_lattice(path, image, optimal_separation, optimal_separation_d , dumbell):
    if (os.path.exists(path+'\\data.hdf5')):
        atom_lattice = am.load_atom_lattice_from_hdf5(path+'\\data.hdf5',construct_zone_axes=False)
    else:
        atom_lattice = get_sublattice(image, optimal_separation, optimal_separation_d , dumbell)
        try:
            atom_lattice.save(path+'\\data.hdf5', overwrite=True)
        except:
            print('Not able to save atom lattice')
    return atom_lattice

def make_intensity_map(path, atom_lattice, images):
    for image_name in images:
        
        if (os.path.exists(path+'\\im_A_'+image_name+'.npy')) & (os.path.exists(path+'\\im_B_'+image_name+'.npy')):
            intensity_A = np.load(path+'\\im_A_'+image_name+'.npy')
            intensity_B = np.load(path+'\\im_B_'+image_name+'.npy')
        else:
            intensity_A, intensity_B = intensity_map(globals()[image_name], atom_lattice)
            intensity_B = intensity_map2(globals()[image_name], atom_lattice)
            np.save(path+'\\im_A_'+image_name+'.npy',intensity_A)
            np.save(path+'\\im_B_'+image_name+'.npy',intensity_B)
            
def make_intensity_map2(path, atom_lattice, images):
    for image_name in images:
        
        if (os.path.exists(path+'\\im_A_'+image_name+'.npy')) & (os.path.exists(path+'\\im_B_'+image_name+'.npy')):
            #intensity_A = np.load(path+'\\im_A_'+image_name+'.npy')
            intensity_B = np.load(path+'\\im_B_'+image_name+'.npy')
        else:
            #intensity_A, intensity_B = intensity_map(globals()[image_name], atom_lattice)
            intensity_B = intensity_map2(globals()[image_name], atom_lattice)
            #np.save(path+'\\im_A_'+image_name+'.npy',intensity_A)
            np.save(path+'\\im_B_'+image_name+'.npy',intensity_B)
    return intensity_B

global pixels_cropped, pixel_size_pm, inner_angle, outer_angle
pixel_size_pm=6.652 #3.326 #6.652 # pm
inner_angle=130
outer_angle=200
pixels_cropped=8

file_paths= open_file()
det_path= open_file()
det_image=hs.signals.Signal2D(plt.imread(det_path[0]))

global dumbell
dumbell = False

for file_path in file_paths:

    et = time.time()
    print(et - st)
    
    s=hs.load(file_path)
    s = s.isig[500:1500, 500:3500]
    
    path = os.path.splitext(file_path)[0]
    if not (os.path.exists(path)):
        os.mkdir(path)

    SL = imp(s,pixels_cropped, pixel_size_pm, inner_angle,outer_angle)
    SL_1 = imp(s,pixels_cropped, pixel_size_pm, inner_angle,outer_angle)
    SL_2 = imp(s,pixels_cropped, pixel_size_pm, inner_angle,outer_angle)
    SL_4 = imp(s,pixels_cropped, pixel_size_pm, inner_angle,outer_angle)
    SL_8 = imp(s,pixels_cropped, pixel_size_pm, inner_angle,outer_angle)
    SL_16 = imp(s,pixels_cropped, pixel_size_pm, inner_angle,outer_angle)
    SL_32 = imp(s,pixels_cropped, pixel_size_pm, inner_angle,outer_angle)
    
    SL.scale(det_image)
    SL_1.scale(det_image)
    SL_2.scale(det_image)
    SL_4.scale(det_image)
    SL_8.scale(det_image)
    SL_16.scale(det_image)
    SL_32.scale(det_image)
    
    SL_2.image.data=SL.PCA(1)
    SL_2.image.data=SL.PCA(2)
    SL_4.image.data=SL.PCA(4)
    SL_8.image.data=SL.PCA(8)
    SL_16.image.data=SL.PCA(16)
    SL_32.image.data=SL.PCA(32)
    
    optimal_separation = 14
    optimal_separation_d = 24
    
    atom_lattice, neighbor_distances = get_sublattice(SL.image, optimal_separation, optimal_separation_d , dumbell)
    atom_lattice_1, neighbor_distances_1 = get_sublattice(SL_1.image, optimal_separation, optimal_separation_d , dumbell)
    atom_lattice_2, neighbor_distances_2 = get_sublattice(SL_2.image, optimal_separation, optimal_separation_d , dumbell)
    atom_lattice_4, neighbor_distances_4 = get_sublattice(SL_4.image, optimal_separation, optimal_separation_d,dumbell)
    atom_lattice_8, neighbor_distances_8 = get_sublattice(SL_8.image, optimal_separation, optimal_separation_d ,dumbell)
    atom_lattice_16, neighbor_distances_16 = get_sublattice(SL_16.image, optimal_separation, optimal_separation_d ,dumbell)
    atom_lattice_32, neighbor_distances_32 = get_sublattice(SL_32.image, optimal_separation, optimal_separation_d,dumbell)
    
    
    neighbor_distance = [neighbor_distances, neighbor_distances_1, neighbor_distances_2, neighbor_distances_4, neighbor_distances_8, neighbor_distances_16, neighbor_distances_32]
    neighbor_distance = np.array(neighbor_distance)
    order = ['Any', 'PCA-1', 'PCA-2', 'PCA-4', 'PCA-8', 'PCA-16', 'PCA-32']
    plt.plot(order, neighbor_distance[:,1], marker='o', linestyle='-')
    #atom_lattice = make_lattice(path, SL_4.image, optimal_separation, optimal_separation_d , dumbell)
    pca_imag = SL_4.image.data
    # Intensity map
    images = ["pca_imag"]
    #make_intensity_map2(path, atom_lattice, images)

    
    #compare(original_imag , pca_imag, 'PCA - 8')
    #intensity_A, intensity_B = intensity_B, intensity_A
    #composition_profile(intensity_A, intensity_B, atom_lattice)
    
    print('end')

    