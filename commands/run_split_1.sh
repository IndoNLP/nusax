models=(bert-base-multilingual-uncased indobenchmark/indobert-base-p1 indolem/indobert-base-uncased xlm-roberta-base w11wo/sundanese-roberta-base)
large_models=(indobenchmark/indobert-large-p1 xlm-roberta-large)

# Sentiment Base
for num_sample in 1 5 10 15 20 -1; do
    for lang in acehnese balinese banjarese buginese english javanese; do
        for ((i = 0; i < ${#models[@]}; ++i)); do
            CUDA_VISIBLE_DEVICES=0 python main.py --task sentiment --dataset $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 0 --num_sample $num_sample &
            CUDA_VISIBLE_DEVICES=1 python main.py --task sentiment --dataset $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 1 --num_sample $num_sample &
            CUDA_VISIBLE_DEVICES=2 python main.py --task sentiment --dataset $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 2 --num_sample $num_sample &
            CUDA_VISIBLE_DEVICES=3 python main.py --task sentiment --dataset $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 3 --num_sample $num_sample &
            CUDA_VISIBLE_DEVICES=4 python main.py --task sentiment --dataset $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 4 --num_sample $num_sample &
            
            wait
            wait
            wait
            wait
            wait

            rm -r save/sentiment/$lang/*/checkpoint*
        done
    done
done

# LID Base
for num_sample in 1 5 10 15 20 -1; do
    for ((i = 0; i < ${#models[@]}; ++i)); do
        CUDA_VISIBLE_DEVICES=0 python main.py --task lid --dataset all --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 0 --num_sample $num_sample &
        CUDA_VISIBLE_DEVICES=1 python main.py --task lid --dataset all --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 1 --num_sample $num_sample &
        CUDA_VISIBLE_DEVICES=2 python main.py --task lid --dataset all --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 2 --num_sample $num_sample &
        CUDA_VISIBLE_DEVICES=3 python main.py --task lid --dataset all --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 3 --num_sample $num_sample &
        CUDA_VISIBLE_DEVICES=4 python main.py --task lid --dataset all --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 4 --num_sample $num_sample &
        
        wait
        wait
        wait
        wait
        wait

        rm -r save/lid/all/*/checkpoint*
    done
done

# LID large
for num_sample in 1 5 10 15 20 -1; do
    for ((i = 0; i < ${#large_models[@]}; ++i)); do
        # LID Large
        CUDA_VISIBLE_DEVICES=0 python main.py --task lid --dataset all --model_checkpoint ${large_models[$i]}--n_epochs 100 --lr 1e-5 --train_batch_size 4 --eval_batch_size 4 --grad_accum 8 --seed 0 --num_sample $num_sample &
        CUDA_VISIBLE_DEVICES=1 python main.py --task lid --dataset all --model_checkpoint ${large_models[$i]}--n_epochs 100 --lr 1e-5 --train_batch_size 4 --eval_batch_size 4 --grad_accum 8 --seed 1 --num_sample $num_sample &
        CUDA_VISIBLE_DEVICES=2 python main.py --task lid --dataset all --model_checkpoint ${large_models[$i]}--n_epochs 100 --lr 1e-5 --train_batch_size 4 --eval_batch_size 4 --grad_accum 8 --seed 2 --num_sample $num_sample &
        CUDA_VISIBLE_DEVICES=3 python main.py --task lid --dataset all --model_checkpoint ${large_models[$i]}--n_epochs 100 --lr 1e-5 --train_batch_size 4 --eval_batch_size 4 --grad_accum 8 --seed 3 --num_sample $num_sample &
        CUDA_VISIBLE_DEVICES=4 python main.py --task lid --dataset all --model_checkpoint ${large_models[$i]}--n_epochs 100 --lr 1e-5 --train_batch_size 4 --eval_batch_size 4 --grad_accum 8 --seed 4 --num_sample $num_sample &

        wait
        wait
        wait
        wait
        wait

        rm -r save/lid/all/*/checkpoint*
    done
done