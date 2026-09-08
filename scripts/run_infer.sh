#!/bin/bash
# 板端批量推理：对 /tmp/rp_in/*.bin 逐帧推理并转储全部输出
# 用法: bash run_infer.sh <model.bin>
MODEL=${1:-~/models/rp0526_split.bin}
OUT=/tmp/rp_out
rm -f $OUT/*.bin $OUT/times.txt
for f in /tmp/rp_in/*.bin; do
  n=$(basename $f .bin)
  rm -f $OUT/model_infer_output_*.bin
  hrt_model_exec infer --model_file $MODEL --input_file $f \
    --enable_dump --dump_format bin --dump_path $OUT > /tmp/rp_last.log 2>&1
  i=0
  for d in $OUT/model_infer_output_*.bin; do
    [ -e "$d" ] || break
    mv $d $OUT/${n}_out$i.bin
    i=$((i+1))
  done
  if [ $i -gt 0 ]; then
    echo "$n: $(grep -i 'infer time' /tmp/rp_last.log) outputs=$i" >> $OUT/times.txt
  else
    echo "$n FAIL: $(tail -1 /tmp/rp_last.log | head -c 120)" >> $OUT/times.txt
  fi
done
cat $OUT/times.txt
