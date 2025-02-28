CUDA_VISIBLE_DEVICES=0 python -u run_kge.py --do_train \
 --cuda \
 --do_test \
 --data_path data/FB15k \
 --model TransE \
 -n 128 -b 1024 -d 200 \
 -g 24.0 -a 1.0 -adv \
 -lr 0.0001 --max_steps 150000 \
 -save ckpts/RotatE_FB15k_0 --test_batch_size 16 
 #-de