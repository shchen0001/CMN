python train.py --dataset cub \
                --gpu-id 2 \
                --concept-dim 2048 \
                --embedding-size 512 \
                --batch-size 250 \
                --lr 7e-4 \
                --warm 8
				--lr-decay-step 5 \
				--bn-freeze 1