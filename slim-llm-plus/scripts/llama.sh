CUDA_VISIBLE_DEVICES=0 python main.py \
--model huggyllama/llama-7b  --eval_ppl \
--epochs 50 --output_dir ./log/llama-7b-w2a16g128 \
--wbits 2 --abits 16 --group_size 128 --lwc --aug_loss

CUDA_VISIBLE_DEVICES=0 python main.py \
--model huggyllama/llama-7b  --eval_ppl \
--epochs 50 --output_dir ./log/llama-7b-w3a16g128 \
--wbits 3 --abits 16 --group_size 128 --lwc --aug_loss

CUDA_VISIBLE_DEVICES=0 python main.py \
--model huggyllama/llama-13b  --eval_ppl \
--epochs 50 --output_dir ./log/llama-13b-w2a16g128 \
--wbits 2 --abits 16 --group_size 128 --lwc --aug_loss

CUDA_VISIBLE_DEVICES=0 python main.py \
--model huggyllama/llama-13b  --eval_ppl \
--epochs 50 --output_dir ./log/llama-13b-w3a16g128 \
--wbits 3 --abits 16 --group_size 128 --lwc --aug_loss


CUDA_VISIBLE_DEVICES=0 python main.py \
--model huggyllama/llama-30b  --eval_ppl \
--epochs 50 --output_dir ./log/llama-30b-w2a16g128 \
--wbits 2 --abits 16 --group_size 128 --lwc --aug_loss

CUDA_VISIBLE_DEVICES=0 python main.py \
--model huggyllama/llama-30b  --eval_ppl \
--epochs 50 --output_dir ./log/llama-30b-w3a16g128 \
--wbits 3 --abits 16 --group_size 128 --lwc --aug_loss

CUDA_VISIBLE_DEVICES=0 python main.py \
--model huggyllama/llama-65b  --eval_ppl \
--epochs 50 --output_dir ./log/llama-65b-w2a16g128 \
--wbits 2 --abits 16 --group_size 128 --lwc --aug_loss

CUDA_VISIBLE_DEVICES=0 python main.py \
--model huggyllama/llama-65b  --eval_ppl \
--epochs 50 --output_dir ./log/llama-65b-w3a16g128 \
--wbits 3 --abits 16 --group_size 128 --lwc --aug_loss