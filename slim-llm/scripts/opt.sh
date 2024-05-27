python run.py \
 facebook/opt-1.3b wikitext2 2bit --groupsize 128 \
--device "cuda:0" --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

python run.py \
 facebook/opt-2.7b wikitext2 2bit --groupsize 128 \
--device "cuda:0" --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

python run.py \
 facebook/opt-6.7b wikitext2 2bit --groupsize 128 \
--device "cuda:0" --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

python run.py \
 facebook/opt-13b wikitext2 2bit --groupsize 128 \
--device "cuda:0" --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

python run.py \
 facebook/opt-30b wikitext2 2bit --groupsize 128 \
--device "cuda:0" --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

python run.py \
 facebook/opt-66b wikitext2 3bit --groupsize 128 \
--device "cuda:0" --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

python run.py \
 facebook/opt-1.3b wikitext2 3bit --groupsize 128 \
--device "cuda:0" --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

python run.py \
 facebook/opt-2.7b wikitext2 3bit --groupsize 128 \
--device "cuda:0" --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

python run.py \
 facebook/opt-6.7b wikitext2 3bit --groupsize 128 \
--device "cuda:0" --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

python run.py \
 facebook/opt-13b wikitext2 3bit --groupsize 128 \
--device "cuda:0" --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

python run.py \
 facebook/opt-30b wikitext2 3bit --groupsize 128 \
--device "cuda:0" --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande


python run.py \
 facebook/opt-66b wikitext2 3bit --groupsize 128 \
--device "cuda:0" --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande
