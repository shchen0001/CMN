python train.py --dataset SOP \
                --gpu-id 0 \
                --concept-dim 2048 \
                --embedding-size 512 \
                --batch-size 150 \
                --lr 7e-4 \
                --warm 1 \
				--lr-decay-step 10 \
				--bn-freeze 0