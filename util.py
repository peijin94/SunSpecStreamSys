from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



def paint_arr_to_jpg(arr, filename='test.jpg', 
                    cmap_name='CMRmap', vmax=None, vmin=None,
                    scaling='linear'):
    """Saves a 2D numpy array as a jpg image.

    Args:
        arr (np.ndarray): The 2D array to save.
        filename (str, optional): The name of the file to save. 
            Defaults to 'test.jpg'.
        do_norm (bool, optional): Whether to normalize the array before saving. 
            Defaults to True.
    """
    if vmax is None:
        vmax = arr.max()
    if vmin is None:
        vmin = arr.min()
    if scaling == 'linear':
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    elif scaling == 'log':
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        raise ValueError(f"Invalid scaling: {scaling}")
    cmap = plt.get_cmap(cmap_name) 
    img = cmap(norm(arr.T))
    imgsave = (img * 255).astype(np.uint8)[:,:,0:3]
    im = Image.fromarray(imgsave)
    im.save(filename)

