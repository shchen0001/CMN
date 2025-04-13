python train.py --dataset cars \
                --gpu-id 0 \
                --concept-dim 2048 \
                --embedding-size 512 \
                --batch-size 150 \
                --lr 7e-4 \
                --warm 8
				--lr-decay-step 5 \
				--bn-freeze 1