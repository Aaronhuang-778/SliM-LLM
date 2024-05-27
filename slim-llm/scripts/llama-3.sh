python run.py \
 meta-llama/Meta-Llama-3-8B wikitext2 2bit --groupsize 128 \
--device "cuda:0" --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

python run.py \
 meta-llama/Meta-Llama-3-8B wikitext2 3bit --groupsize 128 \
--device "cuda:0" --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande
