export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NAME="on"
WRITE="/dockerdata/experiments"
DATA=/apdcephfs/share_916081/brightxwang/data/final_v1

# Reading files
DATA_BIN="$DATA/data_bin"

# Writing files: make sure using $NAME
CHECKPOINT="$WRITE/$NAME/ckpts"
VALID_PATH="$WRITE/$NAME/decoding"
LOGS="$WRITE/$NAME/logs"
DATE=`date +"%b%d-%H-%M"`

SRC=zh
TRG=en
mkdir -p $VALID_PATH
mkdir -p $LOGS

sed -r 's/(@@ )|(@@ ?$)//g' < $DATA/valid.$TRG > $VALID_PATH/valid.$TRG

python train.py \
  $DATA_BIN \
  --ddp-backend=no_c10d \
  --save-dir $CHECKPOINT \
  -s $SRC -t $TRG --fp16 \
  --encoder-layers 40 \
  --encoder-normalize-before \
  --decoder-normalize-before \
  --lr 0.0007 --min-lr 1e-09 \
  --arch san_on_lstm_hybrid \
  --criterion label_smoothed_cross_entropy \
  --chunk-size 128 \
  --weight-decay 0.0 --clip-norm 0.1 --dropout 0.3 \
  --max-tokens 10240 \
  --update-freq 1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 1e-07 \
  --warmup-updates 16000 \
  --label-smoothing 0.1 \
  --no-progress-bar \
  --log-format simple \
  --log-interval 100 \
  --save-interval-updates 1000 \
  --max-update 600000 \
  --max-epoch 100 \
  --beam 1 \
  --remove-bpe \
  --quiet \
  --all-gather-list-size 522240 \
  --num-ref $DATA=1 \
  --valid-decoding-path $VALID_PATH \
  --multi-bleu-path ./scripts/ \
  --save-interval 1 \
  --keep-interval-updates 5 \
  --keep-last-epochs 5 \
  |& tee $LOGS/$DATE.log
