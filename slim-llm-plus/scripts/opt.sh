CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/opt-1.3b --eval_ppl \
--epochs 50 --output_dir ./log/opt-1.3b-w2a16g128 \
--wbits 2 --abits 16 --group_size 128 --lwc \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/opt-1.3b --eval_ppl \
--epochs 50 --output_dir ./log/opt-1.3b-w3a16g128 \
--wbits 3 --abits 16 --group_size 128 --lwc \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/opt-2.7b --eval_ppl \
--epochs 50 --output_dir ./log/opt-2.7b-w2a16g128 \
--wbits 2 --abits 16 --group_size 128 --lwc \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/opt-2.7b --eval_ppl \
--epochs 50 --output_dir ./log/opt-2.7b-w3a16g128 \
--wbits 3 --abits 16 --group_size 128 --lwc \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/opt-6.7b --eval_ppl \
--epochs 50 --output_dir ./log/opt-6.7b-w2a16g128 \
--wbits 2 --abits 16 --group_size 128 --lwc \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/opt-6.7b --eval_ppl \
--epochs 50 --output_dir ./log/opt-6.7b-w3a16g128 \
--wbits 3 --abits 16 --group_size 128 --lwc \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/opt-13b --eval_ppl \
--epochs 50 --output_dir ./log/opt-13b-w2a16g128 \
--wbits 2 --abits 16 --group_size 128 --lwc \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/opt-13b --eval_ppl \
--epochs 50 --output_dir ./log/opt-13b-w3a16g128 \
--wbits 3 --abits 16 --group_size 128 --lwc \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/opt-30b --eval_ppl \
--epochs 50 --output_dir ./log/opt-30b-w2a16g128 \
--wbits 2 --abits 16 --group_size 128 --lwc \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/opt-30b --eval_ppl \
--epochs 50 --output_dir ./log/opt-30b-w3a16g128 \
--wbits 3 --abits 16 --group_size 128 --lwc \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/opt-66b --eval_ppl \
--epochs 50 --output_dir ./log/opt-66b-w2a16g128 \
--wbits 2 --abits 16 --group_size 128 --lwc \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/opt-66b --eval_ppl \
--epochs 50 --output_dir ./log/opt-66b-w3a16g128 \
--wbits 3 --abits 16 --group_size 128 --lwc \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande