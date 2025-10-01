
python3 baselines.py --dataset sst2 --model_checkpoints checkpoints/qlora_llama2_7b_sst2 --unlearn_method gradient_ascent --output_path unlearned_checkpoints/qlora_llama2_7b_sst2_ga --train_batch_size 8 --eval_batch_size 8 --num_epochs 1 --max_length 32
python3 baselines.py --dataset sst2 --model_checkpoints checkpoints/qlora_llama2_7b_sst2 --unlearn_method gradient_ascent --output_path unlearn_checkpoints/ga_llama2_sst2 --train_batch_size 8 --eval_batch_size 8 --num_epochs 1 --max_length 32

python3 baselines.py --dataset sst2 --model_checkpoints checkpoints/qlora_llama2_7b_sst2 --unlearn_method random_label --output_path unlearn_checkpoints/rl_llama2_sst2 --train_batch_size 8 --eval_batch_size 8 --num_epochs 1 --max_length 32

python3 baselines.py --dataset sst2 --model_checkpoints checkpoints/qlora_llama2_7b_sst2 --unlearn_method gradient_ascent_descent --output_path unlearn_checkpoints/gad_llama2_sst2 --train_batch_size 8 --eval_batch_size 8 --num_epochs 1 --max_length 32

python3 baselines.py --dataset sst2 --model_checkpoints checkpoints/qlora_llama2_7b_sst2 --unlearn_method gradient_ascent_kl --logits_path saved_logits/llama-2-7b-hf_sst2.pkl --output_path unlearn_checkpoints/gakl_llama2_sst2 --train_batch_size 8 --eval_batch_size 8 --num_epochs 1 --max_length 32
