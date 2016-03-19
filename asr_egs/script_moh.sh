echo =====================================================================
echo "                        Best Path Decoding                         "
echo =====================================================================

nj=1
max_active=7000
min_active=20
beam=1000.0
beam_delta=1.0
acwt=5.0

# run.pl JOB=1:$nj files/decode.JOB.log \
decode-faster --acoustic-scale=$acwt --allow-partial=true \
    --beam=$beam --beam-delta=$beam_delta --binary=true \
    --max-active=$max_active --min-active=$min_active \
    files/TG.fst \
    ark:files/probs.ark \
    ark,t:files/res_bestpath.ark

echo =====================================================================
echo "                         Lat Gen Decoding                          "
echo =====================================================================

nj=1
max_active=7000
max_mem=50000000
beam=1500.0
lattice_beam=10
acwt=1.0 # acwt: acoustic weight used in getting lattices
acwt_factor=1 

latgen-faster  --max-active=$max_active --max-mem=$max_mem --beam=$beam \
    --lattice-beam=$lattice_beam --acoustic-scale=$acwt \
    --allow-partial=true \
    files/TG.fst \
    ark:files/probs.ark \
    ark:files/lat

lattice-scale --acoustic-scale=$acwt --ascale-factor=$acwt_factor \
    ark:files/lat ark:files/lat_scale.ark 

lattice-best-path ark:files/lat_scale.ark  ark,t:files/res_latgen.ark
