# pvd_morphology
Software for quantitative analysis of microscopy images of *C. elegans* PVD neurons.

*__Note:__ images in this README do not render in the GitHub site. To view a proper rendering, look at `README.html`.*

## Configuration and File Setup
First, clone the repository onto your local computer. On terminal (or Windows Linux subsystem), move to a directory where you would like to download the repo. Once there, run `git clone https://github.com/ajneupauer/pvd_morphology.git` to download the repo. There will now be a folder in your chosen directory called "pvd_morphology". Move into the repo with the command `cd <path_to_parent_directory>/pvd_morphology`.

Once in the repo, set up the Python environment with the command `conda env create -f imaging_env.yml`. If a necessary module is still missing after this step, use `pip install <module_name>` to install it. 

The pipeline expects a certain file structure and naming scheme. Each experiment should have its own directory, with a subdirectory for each image (each image represents a whole PVD neuron). Each raw image file takes the form "{date}\_{description}\_{genotype}\_{age}\_{image #}.tif", where its subdirectory has the same name without the file extension. Below is an example of what an experiment folder should look like once analysis is complete.

experiment_1

* ----| 20260609_pvd_wt_day1_1
  * ----| 20260609_pvd_wt_day1_1_branches.csv
  * ----| 20260609_pvd_wt_day1_1_mip.tif
  * ----| 20260609_pvd_wt_day1_1_mito_rmbg.tif
  * ----| 20260609_pvd_wt_day1_1_mito_seg.tif
  * ----| 20260609_pvd_wt_day1_1_mito_squished.tif
  * ----| 20260609_pvd_wt_day1_1_mito_Straightened.tif
  * ----| 20260609_pvd_wt_day1_1_mito.csv
  * ----| 20260609_pvd_wt_day1_1_nodes.csv
  * ----| 20260609_pvd_wt_day1_1_seg.tif
  * ----| 20260609_pvd_wt_day1_1_squished.tif
  * ----| 20260609_pvd_wt_day1_1_Straightened.tif
  * ----| 20260609_pvd_wt_day1_1_small.tif
  * ----| 20260609_pvd_wt_day1_1.npy
  * ----| 20260609_pvd_wt_day1_1.tif
* ----| 20260609_pvd_wt_day1_2
* ----| stats.csv
* ----| loadings.csv
* ----| pca.png
* ----| report.txt
* ----| plots

Below is an explanation of the different files:

* `*.tif`: raw image. Can also be `.ims`.
* `*squished.tif`: z-stack of the neurite channel, downsampled by 2 in z, y, x.
* `*small.tif`: max intensity projection (MIP) of the neurite channel downsampled by 8 in y, x.
* `*.npy`: coordinates for straightening.
* `*Straightened.tif`: straightened version of `*squished.tif` (z-stack).
* `*mip.tif`: MIP of `*Straightened.tif`.
* `*seg_3d.tif`: 3D mask of the neurites.
* `*seg.tif`: 2D mask of the neurites.
* `*mito_squished.tif`: z-stack of the mitochondrial channel, downsampled by 2 in z.
* `*mito_Straightened.tif`: straightened version of `*mito_squished.tif` (z-stack).
* `*mito_rmbg.tif`: processed MIP of the mitochondrial channel.
* `*mito_seg.tif`: 2D mask of the mitochondrial channel.
* `*branches.csv`: csv file containing branch data on a neuron.
* `*nodes.csv`: csv file containing data on nodes from a network representation of the neurites.
* `*mito.csv`: csv file containing data per mitochondrial foci.
* `stats.csv`: csv file of morphological profiles of all images in the experiment folder.
* `loadings.csv`: principal component loadings of all profile features.
* `pca.png`: 2 component PCA plot of morphological profiles.
* `report.txt`: file with explained variance proportions for PCs #1 and 2 and a list of all images.
* `plots/`: folder storing plots of each feature per age/genotype group.

Finally, the `config.json` file in this repository must be edited to allow the pipeline to work with your specific setup. It can be edited in a basic text editor. Set "has_mito" to 1 if your images have a mitochondrial channel; 0 if they do not. Set "channels" to reflect the order of channels in your images. Set "input_img_fmt" to the file extension of your raw images (".ims" or ".tif"). Set "neurite_seg_path" to the path to the neurite segmentation model. Put "pvd_morphology/ml_models/20250613-pvdseg.pth" unless you are using one you've trained. Set "mito_seg_path" to the path to the mitochondria segmentation model. Put "pvd_morphology/ml_models/20260117-mitoseg.pth" unless you are using one you've trained. Set "classifier_path" to the path to the neurite classification model. Put "pvd_morphology/ml_models/class-3.joblib" unless you are using one you've trained.    

Now that you've downloaded the repo, created the Python environment, organized files, and edited the config, you can proceed to Step 1. More info is provided below on the files in this repository, but it is merely provided for reference.

* `README.html`: this file, as an html.
* `README.md`: this file, as a markdown.
* `config.json`: file specifying parameters that will change depending on the user.
* `generate_coordinates.py`: script opens a Napari interface where the user can collect coordinates down the center of the worm for image straightening. Requires custom module `straightening_utils.py`.
* `image_preprocessing.py`: script takes raw images and produces 'squished' neurite and mitochondria z-stacks, as well as a 'small' 8x downsampled neurite MIP. Requires custom module `ims.py`.
* `imaging_env.yml`: file specifying modules needed for the Python environment for this pipeline.
* `morph_profiling.py`: main program that extracts morphological profiles from straightened images, performs PCA, and plots individual features. Requires custom modules `pvd_plots.py`, `pvd_processing.py`, `models.py`, and `pvd_classifier_1.py`.
* `straighten_3d.py`: script takes 'squished' neurite and mitochondria z-stacks and straightens them using the coordinates saved in the `.npy` file. Requires custom module `straightening_utils.py`.
* `modules/`
    - `branch_reconstructor.py`: module to reconstruct classified neurite fragments into full branches.
    - `ims.py`: module to read ims files.
    - `models.py`: module specifying UNet model architecture for segmentation.
    - `parts.py`: module specifying UNet model architecture for segmentation.
    - `pvd_classifier_1.py`: module specifying random forest model architecture for classification and associated training/prediction methods.
    - `pvd_plots.py`: module for all functions concerning plot generation.
    - `pvd_processing.py`: module for all other functions for PVD image analysis.
    - `straightening_utils.py`: module for image straightening. 
* `ml_models/`
    - `20250613-pvdseg.pth`: most current UNet model for neurite segmentation. 
    - `20260117-mitoseg.pth`: most current UNet model for mitochondria segmentation. 
    - `class-3.joblib`: most current random forest model for neurite classification. 

## Step 1: Image Preprocessing
Always activate your imaging environment before running any commands! Activate it with `micromamba activate imaging`. Move into the repository folder.

Run `image_preprocessing.py` with the following:
```
python ./image_preprocessing.py <path_to_experiment_1> 
```

There will be three outputs per image:

1) An 8x downsampled maximum intensity projection (`*small.tif`, for determination of straightening coordinates)
2) A 2x downsampled neurite z-stack image (`*squished.tif`, to be straightened)
3) A mitochondria z-stack image (`*mito_squished.tif`, to be straightened)

## Step 2: Image Straightening
Always activate your imaging environment before running any commands! Activate it with `micromamba activate imaging`. Move into the repository folder.

Run `generate_coordinates.py` with the following:
```
python ./generate_coordinates.py <path_to_experiment_1> 
```

The shell will prompt you to choose from a list of images. Enter the number of the image you wish to open or press enter to load the first one. A Napari GUI will open for you to extract coordinates down the midline of the worm, which are used to produce straightened images. Follow these six simple steps:

1) Look for a “manual threshold” option on the right-hand widgets. Set it to 105 and check the box for “use manual threshold”. Press “generate mask” to see the results. The mask should more or less follow the contour of the worm, though not perfectly. You may need to play with the threshold and other options like “min size” or “morph open” and regenerate the mask to get a satisfactory result. 
2) Most likely, you will need to edit the mask to precisely outline the boundary of the worm. Do not include the FLP neuron in the mask!
3) Press the “Extract center line” button. If it fails, check for any small groups of pixels separated from the main mask. Erase them and try again. Ensure the line indeed moves down the center of the worm. If it does not, this is a sign to edit your mask and extract the line again. 
4) Finally, press “straighten” to generate a preview of the straightened image. If the posterior end is on the left side, you will need to check the “flip worm” option. The width should also be adjusted so there isn’t much empty space along the top/bottom of the preview. Adjust these settings and re-straighten until achieving the desired result.
5) The results are saved automatically. Note that you are actually generating a list of coordinates down the center of the worm along which the `*squished.tif` images will be straightened. The result is a `.npy` file.
6) Use the dropdown menu on the left to select your next image. Repeat until all images straightening coordinates are completed.

![](/Users/alexneupauer/starr-luxton-lab/pvd-project/pvd_morphology/napari_demo.png)

Once all coordinates are extracted from all images, move onto straightening the images. Run the following:
```
python ./straighten_3d.py <path_to_experiment_1> 
```

There will be two outputs per image:

1) A straightened neurite z-stack image (`*Straightened.tif`)
2) A straightened mitochondria z-stack image (`*mito_Straightened.tif`)

## Step 3: Morphological Profiling
Always activate your imaging environment before running any commands! Activate it with `micromamba activate imaging`. Move into the repository folder.

Run `morph_profiling.py` with the following:
```
python ./morph_profiling.py <path_to_experiment_1> 
```

There will be many outputs: `*mip.tif`, `*seg_3d.tif`, `*seg.tif`, `*mito_squished.tif`, `*mito_Straightened.tif`, `*mito_rmbg.tif`, `*mito_seg.tif`, `*branches.csv`, `*nodes.csv`, `*mito.csv`, `stats.csv`, `loadings.csv`, `pca.png`, `report.txt`, and 131 plots in `plots/`.
