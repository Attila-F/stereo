import os
import numpy as np
import random

from torch.utils.data import Dataset
from scipy import ndimage

# Load random image pair and sample randomly until a valid pixel is found
class KittiLoader(Dataset):
    def __init__(self, datapath, max_disparity, kernel, layers, batch_size):
        # Store the training parameters and list the file folders
        filelist = os.listdir(os.path.join(datapath, 'training', 'disp_noc_1'))
        self.idlist = [filename.split('_')[0] for filename in filelist]

        # Mixed up folders, would be nice to correct but doesnt affect functionality
        self.image_2_template = os.path.join(datapath, 'training', 'image_3', '{}_10.png')
        self.image_3_template = os.path.join(datapath, 'training', 'image_2', '{}_10.png')
        self.labels = os.path.join(datapath, 'training', 'disp_noc_1', '{}_10.png')

        self.max_disparity = max_disparity
        self.receptive_field_size = 2 * (layers * int(kernel/2)) + 1
        self.halfrecp = int(self.receptive_field_size/2)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.idlist)

    def __getitem__(self, id):
        #id = 5
        raw_disparity_image = np.array(ndimage.imread(self.labels.format(repr(id).zfill(6)), mode='I'), dtype=np.float32)

        # Get the actual disparity values
        disparity_image = raw_disparity_image/256

        # Find valid patches and add them to the batch
        valid_patches = 0
        while valid_patches < 1:
            # Entire extended patch must be within the image
            x = random.randint(self.halfrecp, disparity_image.shape[1]-1-self.halfrecp-self.max_disparity)
            y = random.randint(self.halfrecp, disparity_image.shape[0]-1-self.halfrecp)

            local_disparity = int(disparity_image[y,x])
            # Due to 3px loss
            if local_disparity > 2 and local_disparity < self.max_disparity-2:
                valid_patches += 1

                # Load corresponding RGB image patches normalize them and move channels to first dim
                raw_image_2 = np.array(ndimage.imread(self.image_2_template.format(repr(id).zfill(6)), mode='RGB'))
                raw_image_3 = np.array(ndimage.imread(self.image_3_template.format(repr(id).zfill(6)), mode='RGB'))

                image_2 = np.moveaxis((np.uint8(raw_image_2) - 128) / 256, -1, 0)
                image_3 = np.moveaxis((np.uint8(raw_image_3) - 128) / 256, -1, 0)

                # Slice the left image patch around the coordinates and add to batch
                patch_2 = image_2[:, y-self.halfrecp:y+self.halfrecp+1:1, x-self.halfrecp:x+self.halfrecp+1:1]

                patch_3 = image_3[:, y-self.halfrecp:y+self.halfrecp+1:1,
                                     x-self.halfrecp:x+self.max_disparity+self.halfrecp+1:1]
                

        # Return the batches of ground truth and corresponding image patches
        return (local_disparity, patch_2, patch_3)