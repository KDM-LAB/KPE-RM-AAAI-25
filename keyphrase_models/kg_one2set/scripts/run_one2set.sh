#!/bin/bash
home_dir="E:/Backend/one2set/kg_one2set"
# export PYTHONPATH=${home_dir}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0

# data_dir="data/kp20k_separated"

# seed=27
# dropout=0.1
# learning_rate=0.0001
# batch_size=12
# copy_attention=true

# max_kp_len=6
# max_kp_num=20
# loss_scale_pre=0.2
# loss_scale_ab=0.1
# set_loss=true
# assign_steps=2

# model_name="One2set_Copy"
# data_args="Full"
# main_args="Seed27_Dropout0.1_LR0.0001_BS12_MaxLen6_MaxNum20_LossScalePre0.2_LossScaleAb0.1_Step2_SetLoss"

# if [ ${copy_attention} = true ] ; then
#     model_name+="_Copy"
# fi


# if [ "${set_loss}" = true ] ; then
#     main_args+="_SetLoss"
# fi

# save_data="data/kp20k_separated/Full"
# mkdir -p ${save_data}

# exp="Full_One2set_Copy_Seed27_Dropout0.1_LR0.0001_BS12_MaxLen6_MaxNum20_LossScalePre0.2_LossScaleAb0.1_Step2_SetLoss"

# echo "============================= preprocess: ${save_data} ================================="

# preprocess_out_dir="output/preprocess/Full"
# mkdir -p ${preprocess_out_dir}

eval $"cd .."
# cmd="python preprocess.py \
# -data_dir=${data_dir} \
# -save_data_dir=${save_data} \
# -remove_title_eos \
# -log_path=${preprocess_out_dir} \
# -one2many
# "

# echo $cmd
# eval $cmd


# echo "============================= train: ${exp} ================================="

# train_out_dir="output/train/Full_One2set_Copy_Seed27_Dropout0.1_LR0.0001_BS12_MaxLen6_MaxNum20_LossScalePre0.2_LossScaleAb0.1_Step2_SetLoss/"
# mkdir -p ${train_out_dir}

# cmd="python train.py \
# -data ${save_data} \
# -vocab ${save_data} \
# -exp_path ${train_out_dir} \
# -model_path=${train_out_dir} \
# -learning_rate ${learning_rate} \
# -one2many \
# -batch_size ${batch_size} \
# -seed ${seed} \
# -dropout ${dropout} \
# -fix_kp_num_len \
# -max_kp_len ${max_kp_len} \
# -max_kp_num ${max_kp_num} \
# -loss_scale_pre ${loss_scale_pre} \
# -loss_scale_ab ${loss_scale_ab} \
# -assign_steps ${assign_steps} \
# -seperate_pre_ab
# "

# if [ "${copy_attention}" = true ] ; then
#     cmd+=" -copy_attention"
# fi
# if [ "${set_loss}" = true ] ; then
#     cmd+=" -set_loss"
# fi

# echo $cmd
# eval $cmd

# echo "============================= test: ${exp} ================================="

for data in "kp20k"
# for data in "inspec" "krapivin" "nus" "semeval" "kp20k"
do
#   echo "============================= testing ${data} ================================="
#   test_out_dir="output/test/Full_One2set_Copy_Seed27_Dropout0.1_LR0.0001_BS12_MaxLen6_MaxNum20_LossScalePre0.2_LossScaleAb0.1_Step2_SetLoss/kp20k"
#   mkdir -p ${test_out_dir}

#   src_file="data/testsets/kp20k/test_src.txt"
#   trg_file="data/testsets/kp20k/test_trg.txt"

#   -vocab data/kp20k_separated/Full \
#   -src_file=data/testsets/kp20k/test_src.txt \
#   -pred_path output/test/Full_One2set_Copy_Seed27_Dropout0.1_LR0.0001_BS12_MaxLen6_MaxNum20_LossScalePre0.2_LossScaleAb0.1_Step2_SetLoss/kp20k\
#   -exp_path output/test/Full_One2set_Copy_Seed27_Dropout0.1_LR0.0001_BS12_MaxLen6_MaxNum20_LossScalePre0.2_LossScaleAb0.1_Step2_SetLoss/kp20k \
#   -model output/train/Full_One2set_Copy_Seed27_Dropout0.1_LR0.0001_BS12_MaxLen6_MaxNum20_LossScalePre0.2_LossScaleAb0.1_Step2_SetLoss/best_model.pt \
#   -batch_size 20 \
#   -dropout 0.1 \
#   -max_kp_len 6 \
#   -max_kp_num 20 \
#   -seperate_pre_ab -copy_attention
#   -fix_kp_num_len \
#   -remove_title_eos \
#   -one2many \
#   -replace_unk \
  cmd="python predict.py \
  "
#   if [ "$copy_attention" = true ] ; then
#       cmd+=" -copy_attention"
#   fi

#   echo $cmd
  eval $cmd

#   cmd="python evaluate_prediction.py \
#   -pred_file_path ${test_out_dir}/predictions.txt \
#   -src_file_path ${src_file} \
#   -trg_file_path ${trg_file} \
#   -exp_path ${test_out_dir} \
#   -export_filtered_pred \
#   -filtered_pred_path ${test_out_dir} \
#   -disable_extra_one_word_filter \
#   -invalidate_unk \
#   -all_ks 5 M \
#   -present_ks 5 M \
#   -absent_ks 5 M
#   ;cat ${test_out_dir}/results_log_5_M_5_M_5_M.txt
#   "

#   echo $cmd
#   eval $cmd

done

