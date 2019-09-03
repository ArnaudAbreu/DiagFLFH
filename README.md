# DiagFLFH
Code related to submission of "Accurate diagnosis of lymphoma on whole slide histopathology images using deep learning".
# Requirements
* Python >= 3.5.
* Tensorflow >= 1.12.0
* Keras >= 2.2.4
* Scikit-image >= 0.14.12
* Openslide-python >= 1.1.1 (RQ, should install openslide first, please visit https://openslide.org/download/)
* Tqdm >= 4.19.4 (for pretty progressbars)
We mainly tested the code with this configuration, but other versions might work as well.
# Data
Due to heavy disk memory size, Follicular Lymphoma and Follicular Hyperplasia whole slides used in the submission are not provided. Yet, datasets used during the current study are available from the corresponding author on reasonable request.

Nevertheless, the code provided in this repo can be used on user's personal data (FL/FH, or any other multi-class diagnosis task).
# Dataset creation
Given the following file tree structure:

```bash
├── DATA
    ├── CLASS1
    ├── CLASS2
    ├── ...
    ├── CLASSn
        ├── wsifile1.mrxs
        ├── wsifile2.mrxs
        ├── ...
```

One can generate a dataset splitted in "Training", "Validation" and "Test" sets that will be saved as a pickled file on disk in the [DATA] directory under the name `dataset.p` using the following command:

```bash
python dataset_creation.py --data [DATA]
```

After running the above command, we can use the freshly created `dataset.p` to generate patches from WSI files that will be used to train and evaluate the deep learning classifier.

```bash
├── DATASET
    ├── Training
        ├── CLASS1
            ├── patch1.png
            ├── patch2.png
            ├── ...
    ├── Validation
        ├── CLASS1
            ├── patch1.png
            ├── ...
    ├── Test
        ├── CLASS1
            ├── patch1.png
            ├── ...
    ├── Model
```

This is done by running the command:

```bash
python patch_generation.py --datasetfile [DATA]/dataset.p --outputdir [DATASET]
```

Other arguments can be provided to `patch_generation.py` to control the number of png patches, the pixel size of the patches and the resolution pyramid level. use the `--help` option for more information.

The empty folder [Model] will gather later the results from the training of the neural network.
# Fitting the classifier
Training of the deep learning classifier is performed with the script `train_lenet.py` with the command:

```bash
python train_lenet.py --dataset [DATASET]
```

Output of the training is a json-serialized object stored in the [Model] directory created earlier by the `dataset_creation.py` script. This stored trained model can be reloaded later for prediction tasks.
# Testing the classifier
This step consists in storing every prediction of every patch in every test WSI of the dataset. Both mean and variance of the dropout predictions are stored to evaluate classification and uncertainty of the system on each patch.

```bash
python test_lenet.py --dataset [DATASET] --datasetfile [DATA]/dataset.p
```

Results are stored under several `[SLIDENAME].p` pickled files. Each of them is a dictionary with patch identifiers as keys and `(prediction, variance)` as values.
User is then free to read each `[SLIDENAME].p` and compress predictions and variances by averaging on the whole slide for example.
