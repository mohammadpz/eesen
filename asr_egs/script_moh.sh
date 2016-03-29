echo =====================================================================
echo "                        Best Path Decoding                         "
echo =====================================================================

nj=1
max_active=7000
min_active=20
beam=$1
beam_delta=1.0
acwt=$2

decode-faster --acoustic-scale=$acwt --allow-partial=true \
    --beam=$beam --beam-delta=$beam_delta --binary=true \
    --max-active=$max_active --min-active=$min_active \
    --word-symbol-table=/u/zhangy/lisa/eesen/asr_egs/wsj/data/lang_char_larger/words.txt \
    $3 \
    $4 \
    ark,t:$5
