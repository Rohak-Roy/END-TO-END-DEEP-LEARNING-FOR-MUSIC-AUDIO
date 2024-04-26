
STEP 1
-------------------------------------------------------------------------------------

Please note to put the MagnaTagATune folder in the same working directory
as all other files. 


STEP 2
-------------------------------------------------------------------------------------

Before running 'train.py' please run the script 'normalize_data.py'. This file
normalizes the entire dataset.

It creates a new sub-directory under MagnaTagATune called 'samples_norm'.
'samples_norm' will have 2 further sub-directories called 'train' and 'val'.

The model has been trained and evaluated on normalized data so this is a must step.

THINGS TO BE AWARE OF INSIDE 'normalize_data.py':

(1) It requires the path to the training samples.
    This is stored in a variable called 'TRAIN_SAMPLES_PATH' in line 6.
    It's default value is = 'MagnaTagATune/samples/train'

(2) It requires the path to the validation samples.
    This is stored in a variable called 'VAL_SAMPLES_PATH' in line 7.
    It's default value is = 'MagnaTagATune/samples/val'


STEP 3
-------------------------------------------------------------------------------------

The file where the training loop occurs is named 'train.py'. 

THINGS TO BE AWARE OF INSIDE 'train.py':

(1) The parameter LENGTH has to be manually typed in line 16.
    Default Value = 256

(2) The parameter STRIDE has to be manually typed in line 17.
    Default Value = 256

(3) The variable which stores the path to the train labels is called TRAIN_DATA_PATH in line 18. 
    Default Value = 'MagnaTagATune/annotations/train_labels.pkl'

(4) The variable which stores the path to the val labels is called VAL_DATASET_PATH in line 19.
    Default Value = 'MagnaTagATune/annotations/val_labels.pkl'

(5) The variable which stores the path to the normalized samples is called SAMPLES_PATH in line 20. 
    Default Value = 'MagnaTagATune/samples_norm'

(6) If you want to use base model, comment out 'from improved_model import Model' in line 9.
    Uncomment 'from base_model import Model' in line 8.
    When using base model, please change learning from 0.01 to --> 0.001 in optimizer in line 33.
    Please comment out lr_scheduler in line 35 and line 98.

(7) When using improved model, comment out 'from base_model import Model' in line 8.
    Ensure 'from improved_model import Model' is uncommented in line 9.
    Please ensure learning rate is = 0.01 in optimizer in line 33.
    Please ensure lr_scheduler is uncommented in line 35 and line 96.

(8) Everything to do with TensorBoard SummaryWriter is commented out in
    lines 1, 45, 75, 108, 109, 111.
    Please uncomment if required.


STEP 4
-------------------------------------------------------------------------------------

In 'train.py':

(1) In lines 23 and 26 the DataLoaders have an argument num_workers=10.
    For this work the .sh file will require the line #SBATCH --cpus-per-task=10. 

