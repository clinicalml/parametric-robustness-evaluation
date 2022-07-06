# Experiments

This folder contains scripts to reproduce the figures and experiments in our paper.

# Basic Figures 
These figures are straightforward to replicate on a laptop, as they only rely on simple simulations.

## Figure 1 (Section 1)

To produce the loss landscape shown in Figure 1, run
```bash
python experiments/labtest_figures/figure1.py
```
which will output `figs/3d_figure1.pdf`

## Figure 3 (Section 4.1)

To produce the curves shown in Figure 3
```bash
python experiments/labtest_figures/figure3.py
```
which will output `figs/labtest_delta_shift_onlyOL_quad_est.pdf` (the left-hand figure) and `figs/labtest_labtest_shift_onlyOL.pdf` (the right-hand figure)

## Figure 8 (Section C.5)
To produce Figure 8 in Section C.5, run
```bash
python experiments/labtest_ipw_taylor_comparison/illustrative_shifts_get_variance.py
```
and then run `experiments/labtest_ipw_taylor_comparison/labtest_plots.R` in `RStudio`. 

## Figure 9 (Section C.7)
To produce Figure 9 in Section C.7, run
```bash
python experiments/compare_ipw/compare_ipw.py
```
and then run `experiments/compare_ipw/compare_ipw.R` in `RStudio`. 

## Figures 10-11 (Section D)
To produces Figures 10-11 in Section D, run 
```bash
python experiments/labtest_figures/figures10_11.py
```
which will produce 6 figures in total.  The first three are `figs/labtest_subpopulation_worst_case_alpha[0.4,0.6,0.8].pdf`, and make up Figure 10, and the second three are `labtest_user_story_[0,1,2].pdf` and make up Figure 11.

# Celeb A (Section 4.2)

Reproducing the CausalGAN experiment takes a substantial amount of time, primarily in training the original GAN. Note that retraining the GAN will not, in general, reproduce the results in the paper, due to difficulties in controlling randomness in the original CausalGAN code.  However, we also provide the option to use the trained GAN corresponding to the results in the paper.

## Setting up environments

The [CausalGAN](https://github.com/mkocaoglu/CausalGAN) code is (unfortunately) in Python 2, while our code is Python 3. To manage, we have two separate conda environments. To install them, run
```bash
conda env create -f experiments/celeb_gan/conda_environments/environment_CausalGAN.yml
conda env create -f experiments/celeb_gan/conda_environments/environment_shift_gradient.yml
```

## Training the GAN

### Option 1: Use a pre-trained GAN

This is the recommended approach, if you wish to reproduce our results. To unzip and use our pre-trained GAN, run the following.  Note that you will need to have the [kaggle API](https://github.com/Kaggle/kaggle-api) installed, which you can do via `pip install kaggle`.  This is due to a quirk in the original CausalGAN code, which will throw errors if the original data is not available.
```bash
bash experiments/celeb_gan/load_gan.sh
```

### Option 2: Retrain the GAN

We do not recommend doing this, but we include instructions for completeness. First, download the data using the [kaggle API](https://github.com/Kaggle/kaggle-api). 
```bash
cd experiments/celeb_gan
mkdir -p CausalGAN/data/data/celebA
cd CausalGAN/data/data/celebA
kaggle datasets download jessicali9530/celeba-dataset --unzip
cd ../../../../../..
```

Now, to train the CausalGAN model, run the following.  Note that this can take >15hrs on a GPU.
```bash
bash experiments/celeb_gan/train_gan.sh
```

## Training model and evaluating accuracy under shift

The GAN is saved at a `LOAD_PATH` which looks something like `CausalGAN/logs/celebA_0101_010101`. To run the remaining code, update `LOAD_PATH` in [`run.sh`](some_bash_scripts/run.sh), with the timestamp relating to your model.  If you used the recommended "Option 1" above, this is already set to the correct timestamp.

Then a model can be trained, by running
```bash
bash experiments/celeb_gan/run.sh
```
The csv files containing results are saved in `experiments/celeb_gan/results`.

## Producing the tables / figures

Once all of the above has been run, you can produce the tables and figures using the following command.  **NOTE**: You will need to create a new conda environment called `rplots` that has R installed, along with the `tidyverse` and `tikzDevice` packages, and `latex` installed and available on your path.
```bash
bash experiments/celeb_gan/make_tables_figures.sh
```

