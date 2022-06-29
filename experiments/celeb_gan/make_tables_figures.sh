# Enable conda in bash script 
# (https://github.com/conda/conda/issues/7980)
source $(dirname $CONDA_EXE)/../etc/profile.d/conda.sh

export CELEB_PATH='./experiments/celeb_gan'

mkdir -p ${CELEB_PATH}/latex/figures
mkdir -p ${CELEB_PATH}/latex/tables

echo "Generating Tables..."
echo "WARNING: You may need to edit print_results.py due to certain manual calculations that are required for Table 1 (left)"
conda activate shift_gradient
echo ${CELEB_PATH}/print_results.py
python ${CELEB_PATH}/print_results.py > ${CELEB_PATH}/latex/tables/other_results.txt

# To run this, you will need to make a new conda environment that has R
# installed, along with the tidyverse and tikzDevice packages.
# You will also need to have latex installed on your path
echo "Generating Plots... have you installed R and LaTeX?"
conda activate rplots
Rscript ${CELEB_PATH}/plots/figure5_left.R
Rscript ${CELEB_PATH}/plots/figure5_right.R
