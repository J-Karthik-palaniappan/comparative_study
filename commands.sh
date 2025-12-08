python3 run_swinir.py \
    --scale 4 \
    --model_path pretrained_models/SwinIR/x4.pth \
    --folder_lq  datasets/SwinIR/Set5/LR_bicubic/X4 \
    --folder_gt  datasets/SwinIR/Set5/HR \
    --mech       original \
    --device     cuda

# --mech original, pnp, nystrom, performer

# for m in original pnp nystrom performer; do
#   echo "MECH=$m"
#   t=$(date +%s.%N)
#   python run_swinir.py --scale 4 --model_path pretrained_models/SwinIR/x4.pth \
#       --folder_lq datasets/SwinIR/Set5/LR_bicubic/X4 \
#       --folder_gt datasets/SwinIR/Set5/HR \
#       --mech $m --device cuda
#   echo "time=$(echo "$(date +%s.%N) - $t" | bc)s"
#   echo
# done
