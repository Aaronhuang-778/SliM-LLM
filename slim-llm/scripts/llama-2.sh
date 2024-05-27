python run.py \
 ../../Model/llama2/Llama-2-7b-hf wikitext2 2bit --groupsize 128 \
--device "cuda:0" --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

python run.py \
 ../../Model/llama2/Llama-2-13b-hf wikitext2 2bit --groupsize 128 \
--device "cuda:0" --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

python run.py \
 ../../Model/llama2/Llama-2-70b-hf wikitext2 2bit --groupsize 128 \
--device "cuda:0" --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

python run.py \
 ../../Model/llama2/Llama-2-7b-hf wikitext2 3bit --groupsize 128 \
--device "cuda:0" --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

python run.py \
 ../../Model/llama2/Llama-2-13b-hf wikitext2 3bit --groupsize 128 \
--device "cuda:0" --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

python run.py \
 ../../Model/llama2/Llama-2-70b-hf wikitext2 3bit --groupsize 128 \
--device "cuda:0" --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande
