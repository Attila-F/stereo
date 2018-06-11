# Data params
SAVE_PATH = "model/epoch{}.pt"
TEST_EPOCH = 46
RELATIVE_PATH = True
# Expects a folder containing the following structure:
# --training
#    --image_2
#    --image_3
#    --disp_noc_1
# --test
#    --image_2
#    --image_3
KITTIPATH = "/your/absolute/path"

MAX_DISPARITY = 128

# Network params
BATCH_SIZE = 128
CHANNELS = 3
FILTERS = 64
KERNEL = 3
CONV_LAYERS = 4

# Training hyperparameters
LOSS_WEIGHTS = [1/20, 4/20, 10/20, 4/20, 1/20]
MAX_EPOCHS = 300
EPOCH_ITERS = 50
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001