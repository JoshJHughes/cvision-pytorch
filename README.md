# cvision-pytorch

This is a collection of small projects I used to learn/practice PyTorch and 
computer vision

## Installation

To install the package navigate to its root directory and use the command

> pip install .

Please note that this will install the cpu only build of pytorch and 
torchvision, to use a cuda build please install it yourself prior to running 
`pip install .`

## Project Structure

Run the code found in `scripts/`.  It calls the packages from `src/` - my code 
can be found in `cvision/`, `references/` contains code from the torchvision 
tutorials.  

All code expects the datasets to be stored in `data/` under the project root 
folder.  I recommend creating a symbolic link to your preferred location.  