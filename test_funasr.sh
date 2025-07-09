export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
model_dir='./pretrain/paraformer-zh'
input='/data/zhanglingling/AI/wenet/examples/aishell/s0/data/test/data.list'
output='/data/zhanglingling/AI/wenet/result_paraformer-zh/text'
hotwords='/data/zhanglingling/AI/wenet/hotwords/hotwords_aishell.txt'
verbose=true 
batch_size=16
if [ "$verbose" = "true" ]; then
  verbose_flag="--verbose"
else
  verbose_flag=""
fi
python test_funasr.py \
    --input $input \
    --output $output \
    --model $model_dir $verbose_flag \
    --batch_size $batch_size \
    #--hotwords $hotwords \
    