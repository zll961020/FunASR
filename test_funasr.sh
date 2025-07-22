export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
model_dir='./pretrain/SeACoParaformer-zh'
input='/data/zhanglingling/AI/wenet/examples/aishell/s0/data/test/data.list'
mode='beam_search'
result_dir='/data/zhanglingling/AI/wenet/result_seaco_paraformer-zh_beam_search/'$mode
output=$result_dir'/text'
hotwords='/data/zhanglingling/AI/wenet/hotwords/hotwords_aishell.txt'
verbose=true 
batch_size=16
context_graph_score=0.0
decoding_ctc_weight=0.4
beam_size=5 

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
    --hotwords $hotwords \
    --context_graph_score $context_graph_score \
    --decoding_ctc_weight $decoding_ctc_weight \
    --beam_size $beam_size \
    --mode $mode \

cp "$0" "$result_dir/"
    