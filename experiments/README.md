# Experiment: Celeb A (Section 4.2)
We here detail a number of experiments from the paper [Evaluating Robustness to Dataset Shift
via Parametric Robustness Sets](). 

## Downloading data
First, download the data using the [kaggle API](https://github.com/Kaggle/kaggle-api). 
```bash
$ mkdir -p CausalGAN/data/data/celebA
$ cd CausalGAN/data/data/celebA
$ kaggle datasets download jessicali9530/celeba-dataset --unzip
$ cd ../../../..
```

The CausalGAN code (unfortunately) in Python 2, while our code is Python 3. To manage, we have two separate conda environments. To install them, run
```bash
$ conda env create -f conda_environments/environment_CausalGAN.yml
$ conda env create -f conda_environments/environment_shift_gradient.yml
```

## Fitting GAN
Now, to train the CausalGAN model, run
```bash
$ conda activate CausalGAN
$ cd CausalGAN
$ python main.py --causal_model big_causal_graph --is_pretrain True --model_type began --is_train True --num_iter 250000
```


## Training model and evaluating accuracy under shift
The GAN is saves at a `LOAD_PATH` which looks something like `CausalGAN/logs/celebA_0101_010101`. To run the remaining code, update `LOAD_PATH` in [`run.sh`](some_bash_scripts/run.sh), with the timestamp relating to your model. 

Then a model can be trained, by running
```bash
$ bash run.sh
```
The csv files containing results are saved in `experiments/celeb_gan/results/default_experiment` (unless `FOLDER_NAME` is changed in `run.sh`.)


# Experiment: Compare IPW and Taylor variance in labtest example (Section C.5)
To produce Figure 8 in Section C.5, run
```bash
$ experiments/labtest_ipw_taylor_comparison/illustrative_shifts_get_variance.py
```
and then run `experiments/labtest_ipw_taylor_comparison/labtest_plots.R` in `RStudio`. 


# Experiment: Compare IPW and Taylor estimation (Section C.7)
To produce Figure 9 in Section C.7, run
```bash
$ python experiments/compare_ipw/compare_ipw.py
```
and then run `experiments/compare_ipw/compare_ipw.R` in `RStudio`. 