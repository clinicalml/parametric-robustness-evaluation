# Enable conda in bash script 
# (https://github.com/conda/conda/issues/7980)
source $(dirname $CONDA_EXE)/../etc/profile.d/conda.sh

export LOAD_PATH="./logs/celebA_0527_120632"

# Train model on simulated images
conda activate CausalGAN
cd experiments/celeb_gan/CausalGAN  
python generate_training_data.py --causal_model big_causal_graph --load_path $LOAD_PATH --model_type 'began'
cd ../../..

conda activate shift_gradient
python experiments/celeb_gan/train_model.py --num_epochs 25

# ############################################################################
# Figure 5 (Right): Simulate training data, estimate worst-case directions 
# using Taylor and IPW, simulate data from worst-case direction, and evaluate
# ############################################################################

# Simulate 100 training distributions
conda activate CausalGAN
cd experiments/celeb_gan/CausalGAN  
python generate_ipw_taylor_comparison.py --n_sims 100 --causal_model big_causal_graph --load_path $LOAD_PATH --model_type 'began' --M 2
cd ../../..

# For each training dataset, estimate worst-case direction using Taylor and IPW
conda activate shift_gradient
python experiments/celeb_gan/estimate_worst_case_shifts.py --n_sims 100

# Simulate data from worst-case directions
conda activate CausalGAN
cd experiments/celeb_gan/CausalGAN  
python generate_ipw_taylor_comparison.py --load_deltas True --causal_model big_causal_graph --load_path $LOAD_PATH --model_type 'began' --M 10
cd ../../..

# Evaluate data from worst-case directions
conda activate shift_gradient
python experiments/celeb_gan/evaluate_worst_case_shifts.py

# ############################################################################
# Figure 5 (Left): Simulate random shifts and compare to the worst-case shift
# identified by the taylor approximation
# ############################################################################

# Simulate random shifts and corresponding set of images
conda activate CausalGAN
cd experiments/celeb_gan/CausalGAN  
python generate_random_shift_data_31_dim.py --causal_model big_causal_graph --load_path $LOAD_PATH --model_type 'began' --n_random 400
cd ../../..

# Combine simulated test with estimated losses
conda activate shift_gradient
python experiments/celeb_gan/evaluate_random_shifts.py --use_31 True --n_random 400