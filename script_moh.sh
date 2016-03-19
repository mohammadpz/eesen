echo =====================================================================
echo "                        Best Path Decoding                         "
echo =====================================================================

nj=1
max_active=7000
min_active=20
max-mem=500000
beam=15.0
beam_delta=0.5
acoustic_scale=0.5

run.pl JOB=1:$nj files/decode.JOB.log \
decode-faster --acoustic-scale=$acoustic_scale --allow-partial=true \
    --beam=$beam --beam-delta=$beam_delta --binary=true \
    --max-active=$max_active --min-active=$min_active \
    --word-symbol-table=words.txt \
    files/L.fst \
    ark:files/log_probs.ark \
    ark,t:files/res.ark

echo =====================================================================
echo "                            Scoring                                "
echo =====================================================================

cat files/text | sed 's:<UNK>::g' | sed 's:<NOISE>::g' | \
    sed 's:<SPOKEN_NOISE>::g' > files/text_filt 
cat res.ark | utils/int2sym.pl -f 2- files/words.txt | \
  compute-wer --text --mode=present ark:files/text_filt ark,p:-  >& files/wer

echo =====================================================================
echo "                      Beam Search Decoding                         "
echo =====================================================================

nj=1
max_active=7000
max-mem=500000
beam=15.0
lattice_beam=10
acoustic_scale=0.5

run.pl JOB=1:$nj files/decode.JOB.log \
latgen-faster  --max-active=$max_active --max-mem=$max_mem --beam=$beam
    --lattice-beam=$lattice_beam --acoustic-scale=$acoustic_scale \
    --allow-partial=true \
    --word-symbol-table=files/words.txt \
    files/L.fst \
    ark:files/log_probs.ark \
    "ark:|gzip -c > files/lat.JOB.gz"

echo =====================================================================
echo "                            Scoring                                "
echo =====================================================================

min_acwt=5
max_acwt=10
acwt_factor=0.1
data_dir=files
lm_dir=files
decode_dir=files

score.sh --min-acwt $min_acwt --max-acwt $max_acwt --acwt-factor \
    $acwt_factor --cmd "run.pl" $data_dir $lm_dir $decode_dir
