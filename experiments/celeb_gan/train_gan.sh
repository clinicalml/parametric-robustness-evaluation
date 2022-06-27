# Enable conda in bash script 
# (https://github.com/conda/conda/issues/7980)
source $(dirname $CONDA_EXE)/../etc/profile.d/conda.sh

conda activate CausalGAN
cd experiments/celeb_gan/CausalGAN
python main.py --causal_model big_causal_graph --is_pretrain True --model_type began --is_train True --num_iter 250000 --seed 22
