# pvd_morphology
Software for quantitative analysis of microscopy images of *C. elegans* PVD neurons.

## Configuration and File Setup
To set up the Python environment, use the `imaging_env.yml` file in this repo to create an environment. If some necessary modules are still missing after this step, use `pip install`. To load the custom modules in this repo, run:
```
import sys
sys.path.append('{modules_dir}')
```
Then modules can simply be imported with `import {custom_module}`.

Here is a table of all the Python files to run in this pipeline and their required modules:
| Script | Required Modules |
| ---- | ---- |
| image_preprocessing.py | tifffile, pathlib.Path, numpy, ims.ImarisReader[^1] |
| generate_coordinates.py | pathlib.Path, numpy, scipy.ndimage, skimage, tifffile, magicgui.magicgui, napari.layers.Image, napari.layers.Labels, napari.layers.Shapes, napari.types.ImageData, napari.types.LabelsData, napari.types.LayerDataTuple, scipy.interpolate, scipy.optimize.minimize, straightening_utilsº | 
| image_to_branches.py | tifffile, pathlib.Path, torch, numpy, pvd_processingº, modelsº, pvd_classifier_1º |
| feature_extraction.py | pandas, tifffile, pathlib.Path, seaborn, matplotlib.pyplot, numpy, scipy.stats, sklearn.preprocessing.StandardScaler, sklearn.decomposition.PCA, pvd_plotsº |

[^1]: Custom module in this repo.

All scripts manage module imports, but this table is useful as reference.

The recommended file structure consists of a typical subdirectories contained within a directory for each separate experiment. Within each directory are the following subdirectories:
* branches: csv files containing branch data per each neuron, named as `*branches.csv`.
* featureExtraction: plots and PCA outputs go here.
* maxProj: max intensity projections of 2x downsampled straightened images, named as `*maxProj.tif`.
* segmentations: masks of neuron images, named as `*seg.tif`.
* straightenedImgs: 2x downsampled straightened images, named as `*Straightened.tif`.

Original images, their coordinates, and downsampled versions can reside within the directory as loose files (not in a subdirectory). This format does not need to be followed but is highly recommended, as different file organizations may require adjustments of the scripts.

## Step 1: Obtain Straightening Coordinates
Run `image_preprocessing.py` to perform preprocessing of images. There will be two outputs:
1) An 8x downsampled maximum intensity projection (`*small.tif`, for determination of straightening coordinates)
2) A 2x downsampled z-stack image (`*squished.tif`, to be straightened)

*Note that parts of the script may need to be adjusted to match your file organization structure!*

Run `generate_coordinates.py` to open a Napari GUI. Set line 204 to the first 8x downsampled file:
```
path = Path("{path_to_your_8xds_image}")
```

The GUI will extract coordinates down the midline of the worm, which are used to produce straightened images. Follow these six simple steps:
1) Blur the 8x downsampled image. Adjust the 'sigma' blur radius value as desired and click 'Run' in the upper right corner.
2) In the second sidebar module, select the image to blur as 'blurred (data)' and click 'generate mask'.
3) The mask is meant to cover the area occupied by PVD, but it will likely need to be manually adjusted at this stage in software development. Use the Napari eraser andn paintbrush tools to modify the mask layer. Watch for regions of PVD not in the mask (add), non PVD areas in the mask (delete), and regions of FLP in the mask (delete).
4) In the third sidebar module, select the label as 'threshold_image_result (data)' and click 'Extract center line'. If an error occurs, it is likely due to the mask having more than one continuous blob or 'holes' in the mask.
5) In the fourth sidebar module, make sure the image layer is set to 'image (data)' and the path layer is set to 'center line'. For the 'spline output' click select file to name the output file for the coordinates.
6) To save the coordinates, press 'straighten'. A preview of the straightened image will appear in the GUI. If the anterior-posterior orientation is incorrect, check 'flip worm'. If the thickness needs to be adjusted, change the 'width'. Press 'straighten' again to get the coordinates with the altered settings.

![]("/Users/alexneupauer/Desktop/Screenshot 2025-10-07 at 4.10.05 PM.png")

While the GUI is open, all layers can be deleted once straightening is complete and the next 8x downsampled image can be dragged and dropped into the GUI. Repeat the same steps on this next image.

## Step 2: Straighten, Mask, and Classify Dendrites
Run `image_to_branches.py` to straighten each image, make a mask, classify the branches as 1º - 4º, and compute statistics on branches. It is critical to ensure the paths to the segmentation and classification models are correct.

*Note that parts of the script may need to be adjusted to match your file organization structure!*

## Step 3: Morphological Profiling
Running the functions in `feature_extraction.py` will produce morphological profiles of each worm image, perform PCA on those profiles, and plot specific morphological features. To generate morphological profiles, use:
```
data = write_feature_table('{dir_of_your_*branches.csv_files_from_step2}')
data.to_csv('{destination_to_save_profiles}')
```
The final cell provides codes for PCA. A 2D plot of the first two PCs can be generated. The explained variance contribution of each PC and the feature loadings for each PC can also be computed.

## Step 4: Comparing Single Features
The PCA will compare high-dimensional morphological profiles consisting of many features, such as the counts and cumulative lengths of different dendrite types. To plot a stripchart comparing multiple groups of worms (genotype and age) for a specific feature, use the following command in `feature_extraction.py`:
```
pvd_plots.phenotype_stripchart(data, '{feature_of_interest}', {tuple_of_plot_dimensions_inches})
```


 
