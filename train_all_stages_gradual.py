import os

"""
We manage all training and test commands in this file

This pipeline is easy to understand:

1.From cityscapes day to dark zurich day
  -- Generate aligned model - day_to_day
  -- Generate pseudo label - gen_day_pseudo
  -- Finetune with pseudo - ft_day_pseudo

2.From day to twilight
  -- Generate rectified model - day_to_twilight
  -- Generate pseudo label - gen_twilight_pseudo
  -- Finetune with pseudo - ft_twilight_pseudo

3.From twilight to night
  -- Generate rectified model - twilight_to_night
  -- Generate pseudo label - gen_night_pseudo
  -- Finetune with pseudo - ft_night_pseudo

4.Test on dark zurich night - gen_test_result
"""

gpu_ids = '0'
input_size = '1024,512'
crop_size = '1024,512'
ft_step = 5000
outdir = 'gradual'

# --------------------------- Base args ---------------------------
# Align base args for first step
align_args = 'python train_ms.py  --tensorboard  \
--drop 0.1 --batch-size 2 --iter-size 1 --lambda-seg 0.5  --lambda-adv-target1 \
0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn  --class-balance  \
--only-hard-label -1  --max-value 7  --often-balance  --use-se  --save-pred-every 2500 \
--crop-size %s --input-size %s --input-size-target %s --gpu-ids %s ' % (crop_size, input_size, input_size, gpu_ids)
# Rectify base args for second and third step
rectify_args = align_args + ' --learning-rate 1e-5  --kl-warm-up 0 --num-steps %s ' % (ft_step)
# Fintune base args
ft_args = 'python train_ft.py --tensorboard \
--droprate 0.2 --warm-up 500 --batch-size 6 --learning-rate 1e-4 --crop-size 512,256 --lambda-seg 0.5 \
--lambda-adv-target1 0 --lambda-adv-target2 0 --lambda-me-target 0 --lambda-kl-target 0 --norm-style gn \
--class-balance --only-hard-label -1 --max-value 7 --often-balance  --use-se \
--input-size 1024,612  --train_bn  --autoaug False  --num-steps %s  --save-pred-every 2500 --gpu-ids %s \
' % (ft_step, gpu_ids)

# --------------------------- Cross Domain Training Commands ---------------------------
day_to_day = align_args + ' --learning-rate 2e-4 --warm-up 500 --kl-warm-up 5000 --num-steps 20000 --snapshot-dir ./snapshots/%s/day_to_day --set day ' % (outdir)
day_to_twilight = rectify_args + ' --snapshot-dir ./snapshots/%s/day_to_twilight --set twilight  --restore-from ./snapshots/%s/ft_day_pseudo/GTA5_%s.pth ' % (outdir, outdir, ft_step)
twilight_to_night = rectify_args + ' --snapshot-dir ./snapshots/%s/twilight_to_night --set night --restore-from ./snapshots/%s/ft_twilight_pseudo/GTA5_%s.pth ' % (outdir, outdir, ft_step)

# --------------------------- Pseudo Generation Commands ---------------------------
gen_day_pseudo = 'python generate_plabel_dark_zurich.py --set day \
--restore-from ./snapshots/%s/day_to_day/GTA5_20000.pth --save ./data/%s/day --input-size %s ' % (outdir, outdir, input_size)
gen_twilight_pseudo = 'python generate_plabel_dark_zurich.py --set twilight \
--restore-from ./snapshots/%s/day_to_twilight/GTA5_%s.pth --save ./data/%s/twilight --input-size %s ' % (outdir, ft_step, outdir, input_size)
gen_night_pseudo = 'python generate_plabel_dark_zurich.py --set night \
--restore-from ./snapshots/%s/twilight_to_night/GTA5_%s.pth --save ./data/%s/night --input-size %s ' % (outdir, ft_step, outdir, input_size)
gen_val_result = 'python generate_plabel_dark_zurich.py --set val \
--restore-from ./snapshots/%s/ft_night_pseudo/GTA5_%s.pth --save ./data/%s/val --input-size %s ' % (outdir, ft_step, outdir, input_size)
gen_test_result = 'python generate_plabel_dark_zurich.py --set test \
--restore-from ./snapshots/%s/ft_night_pseudo/GTA5_%s.pth --save ./data/%s/test --input-size %s ' % (outdir, ft_step, outdir, input_size)

# --------------------------- Fintune Commands ---------------------------
ft_day_pseudo = ft_args + ' --snapshot-dir ./snapshots/%s/ft_day_pseudo --restore-from ./snapshots/%s/day_to_day/GTA5_20000.pth --set %s/day' % (outdir, outdir, outdir)
ft_twilight_pseudo = ft_args + ' --snapshot-dir ./snapshots/%s/ft_twilight_pseudo --restore-from ./snapshots/%s/day_to_twilight/GTA5_%s.pth --set %s/twilight' % (outdir, outdir, ft_step, outdir)
ft_night_pseudo = ft_args + ' --snapshot-dir ./snapshots/%s/ft_night_pseudo --restore-from ./snapshots/%s/twilight_to_night/GTA5_%s.pth --set %s/night ' % (outdir, outdir, ft_step, outdir)

# --------------------------- All Commands ---------------------------
all_commands = [day_to_day, gen_day_pseudo, ft_day_pseudo, day_to_twilight, gen_twilight_pseudo, ft_twilight_pseudo, twilight_to_night, gen_night_pseudo, ft_night_pseudo, gen_val_result, gen_test_result]

for command in all_commands:
    print(command)
    os.system(command)
