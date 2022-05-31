# IMPORTANT: You may need to enable conda in bash script (https://github.com/conda/conda/issues/7980#issuecomment-441358406)
source /opt/conda/thams/etc/profile.d/conda.sh
export FOLDER_NAME="default_experiment"
export LOAD_PATH="./logs/celebA_0527_120632"



# # Train model on simulated images
# cd CausalGAN
# conda activate CausalGAN
# python generate_training_data.py --causal_model big_causal_graph --load_path $LOAD_PATH --model_type 'began'
# cd ..
# conda activate shift_gradient
# python experiments/celeb_gan/train_model.py --folder_name $FOLDER_NAME --num_epochs 25


## Simulate training data, estimate worst-case directions using Taylor and IPW, simulate data from worst-case direction, and evaluate (Figure 5, right)
# Simulate training data
cd CausalGAN
conda activate CausalGAN
python generate_ipw_taylor_comparison.py --n_sims 100 --causal_model big_causal_graph --load_path $LOAD_PATH --model_type 'began' --M 2

# For each training dataset, estimate worst-case direction using Taylor and IPW
conda activate shift_gradient
cd ..
python experiments/celeb_gan/estimate_worst_case_shifts.py --n_sims 100

# Simulate data from worst-case directions
cd CausalGAN
conda activate CausalGAN
python generate_ipw_taylor_comparison.py --load_deltas True --causal_model big_causal_graph --load_path $LOAD_PATH --model_type 'began' --M 10

# Evaluate data from worst-case directions
conda activate shift_gradient
cd ..
python experiments/celeb_gan/evaluate_worst_case_shifts.py

## Simulate random shifts and compare to the worst-case shift identified (Figure 5, left)
# Simulate random test environments
cd CausalGAN
conda activate CausalGAN
python generate_random_shift_data_31_dim.py --causal_model big_causal_graph --load_path $LOAD_PATH --model_type 'began' --n_random 400

# Combine simulated test with estimated losses
conda activate shift_gradient
cd ..
python experiments/celeb_gan/evaluate_random_shifts.py --use_31 True --n_random 400
