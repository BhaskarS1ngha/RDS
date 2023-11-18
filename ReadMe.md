# Rough Denisty Segmentation
## Introduction
This repository contains the  implementation of the method described in the paper *[Jitani, Nitya & Singha, Bhaskar & Barman, Geetanjali & Talukdar, Abhijit & Sarmah, Rosy & Bhattacharyya, Dhruba K. (2023). Medical image segmentation using automated rough density approach. Multimedia Tools and Applications. 1-29. 10.1007/s11042-023-16921-6.](http://dx.doi.org/10.1007/s11042-023-16921-6)*

## Requirements
>OS: Linux (Ubuntu 20.04)\
>Python: 3.8.5\
>Packages:
>* numpy
>* opencv-python
>* pillow
>* matplotlib\
>**Note:** Detailed version information can be found in the [requirements.txt](requirements.txt) file.

## Usage

```bash
# Clone the repository
git clone https://github.com/BhaskarS1ngha/RDS.git
cd RDS

# 2. Install the requirements
pip install -r requirements.txt

# 3. Run the code

python rough_density_segmentation.py <path_to_image> <threshold>
# Threshold is optional and defaults to 70
```

## Integrating with your code
The file [rough_density_segmentation.py](rough_density_segmentation.py) contains the function 'cluster' which can be used to segment an image. The function takes two arguments: \
>IMAGE_PATH: Path to the image to be segmented \
>THRESHOLD: Threshold value for the rough density segmentation. Defaults to 70 if not provided.
> 
The function returns a numpy array containing the segmented image.

## Miscellaneous
The folder [SampleData](SampleData) contains some sample X-Ray images of human lungs used for testing the code.




