export CUDA_VISIBLE_DEVICES=4
python3 training.py --train_ds=df_random_nd_c_train --val_ds=df_random_nd_c_val --test_ds=df_quebec_nd_c_test --mode=test
