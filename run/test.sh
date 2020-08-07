NAME=transformer-de-en-big

CHECKPOINT=/dockerdata/experiments/$NAME/ckpts
DATA_BIN=/apdcephfs/share_916081/ychao/data/wmt_en_de_stanford/joined
ckpt=checkpoint_best.pt
DECODING=/dockerdata/experiments/$NAME/decoding/test
mkdir -p $DECODING

python generate.py \
    $DATA_BIN \
    -s de -t en \
    --gen-subset train \
    --task translation \
    --path $CHECKPOINT/$ckpt \
    --lenpen 0.6 \
    --beam 4 \
    --valid-decoding-path $DECODING \
    --decoding-path $DECODING \
    --multi-bleu-path ./scripts \
    --num-ref $DATA=1 \
    |& tee $DECODING/$ckpt.gen
