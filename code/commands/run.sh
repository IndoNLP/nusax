models=(bert-base-multilingual-uncased indobenchmark/indobert-base-p1 indolem/indobert-base-uncased)

# Sentiment Base

for num_sample in 1 5 10 15 20 -1; do
    for lang in acehnese balinese banjarese buginese english javanese maduranese minangkabau ngaju sundanese toba_batak all; do
        for ((rand = 0; rand < 5; ++rand)); do
            for ((i = 0; i < ${#models[@]}; ++i)); do
                CUDA_VISIBLE_DEVICES=$i python main.py --task sentiment --dataset $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed $rand --num_sample $num_sample &
            done
            CUDA_VISIBLE_DEVICES=3 python main.py --task sentiment --dataset $lang --model_checkpoint xlm-roberta-base --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed $rand --num_sample $num_sample
            wait
            rm -r save/sentiment/$lang/*/checkpoint*
        done
    done
done

# LID Base
for num_sample in 1 5 10 15 20; do
    for ((rand = 0; rand < 5; ++rand)); do
        for ((i = 1; i < ${#models[@]}; ++i)); do
            CUDA_VISIBLE_DEVICES=$i python main.py --task lid --dataset all --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed $rand --num_sample $num_sample &
        done
        CUDA_VISIBLE_DEVICES=3 python main.py --task lid --dataset all --model_checkpoint xlm-roberta-base --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed $rand --num_sample $num_sample
        wait
        rm -r save/lid/$lang/*/checkpoint*
    done
done

# LID & Sentiment large
for num_sample in 1 5 10 15 20; do
    for ((rand = 0; rand < 5; ++rand)); do
        # LID Large
        CUDA_VISIBLE_DEVICES=0 python main.py --task lid --dataset all --model_checkpoint indobenchmark/indobert-large-p1 --n_epochs 100 --lr 1e-5 --train_batch_size 4 --eval_batch_size 4 --grad_accum 8 --seed $rand --num_sample $num_sample &
        CUDA_VISIBLE_DEVICES=1 python main.py --task lid --dataset all --model_checkpoint xlm-roberta-large --n_epochs 100 --lr 1e-5 --train_batch_size 4 --eval_batch_size 4 --grad_accum 8 --seed $rand --num_sample $num_sample 

        # Sentiment All Large
        CUDA_VISIBLE_DEVICES=2 python main.py --task sentiment --dataset all --model_checkpoint indobenchmark/indobert-large-p1 --n_epochs 100 --lr 1e-5 --train_batch_size 4 --eval_batch_size 4 --grad_accum 8 --seed $rand --num_sample $num_sample &
        CUDA_VISIBLE_DEVICES=3 python main.py --task sentiment --dataset all --model_checkpoint xlm-roberta-large --n_epochs 100 --lr 1e-5 --train_batch_size 4 --eval_batch_size 4 --grad_accum 8 --seed $rand --num_sample $num_sample
        wait

        rm -r save/sentiment/$lang/*/checkpoint*
        rm -r save/lid/all/*/checkpoint*

        for lang in acehnese balinese banjarese buginese english javanese maduranese minangkabau ngaju sundanese toba_batak all; do
            CUDA_VISIBLE_DEVICES=0 python main.py --task sentiment --dataset $lang --model_checkpoint indobenchmark/indobert-large-p1 --n_epochs 100 --lr 1e-5 --train_batch_size 4 --eval_batch_size 4 --grad_accum 8 --seed $rand --num_sample $num_sample &
            CUDA_VISIBLE_DEVICES=3 python main.py --task sentiment --dataset $lang --model_checkpoint xlm-roberta-large --n_epochs 100 --lr 1e-5 --train_batch_size 4 --eval_batch_size 4 --grad_accum 8 --seed $rand --num_sample $num_sample
            wait
            rm -r save/sentiment/$lang/*/checkpoint*
    done
done

####
# Untrained Model
####

# for num_sample in 1 5 10 15 20 -1; do
#     for ((rand = 0; rand < 5; ++rand)); do
#         CUDA_VISIBLE_DEVICES=0 python main.py --task sentiment --dataset aceh --model_checkpoint none --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed $rand --num_sample $num_sample &
#         CUDA_VISIBLE_DEVICES=1 python main.py --task sentiment --dataset bali --model_checkpoint none --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed $rand --num_sample $num_sample &
#         CUDA_VISIBLE_DEVICES=2 python main.py --task sentiment --dataset banjar --model_checkpoint none --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed $rand --num_sample $num_sample &
#         CUDA_VISIBLE_DEVICES=3 python main.py --task sentiment --dataset bugis --model_checkpoint none --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed $rand --num_sample $num_sample
#         wait
#         CUDA_VISIBLE_DEVICES=0 python main.py --task sentiment --dataset jawa --model_checkpoint none --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed $rand --num_sample $num_sample &
#         CUDA_VISIBLE_DEVICES=1 python main.py --task sentiment --dataset madura --model_checkpoint none --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed $rand --num_sample $num_sample &
#         CUDA_VISIBLE_DEVICES=2 python main.py --task sentiment --dataset minang --model_checkpoint none --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed $rand --num_sample $num_sample &
#         CUDA_VISIBLE_DEVICES=3 python main.py --task sentiment --dataset ngaju --model_checkpoint none --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed $rand --num_sample $num_sample
#         wait
#         CUDA_VISIBLE_DEVICES=0 python main.py --task sentiment --dataset sunda --model_checkpoint none --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed $rand --num_sample $num_sample &
#         CUDA_VISIBLE_DEVICES=1 python main.py --task sentiment --dataset all --model_checkpoint none --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed $rand --num_sample $num_sample &
#         CUDA_VISIBLE_DEVICES=3 python main.py --task lid --dataset all --model_checkpoint none --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed $rand --num_sample $num_sample
#         wait
#         rm -r save/sentiment/$lang/*/checkpoint*
#         rm -r save/lid/$lang/*/checkpoint*        
#     done        
# done