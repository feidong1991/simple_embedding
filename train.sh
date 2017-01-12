# @Author: feidong
# @Date:   2017-01-03 16:07:32
# @Last Modified by:   feidong
# @Last Modified time: 2017-01-12 20:53:47

datadir=./data
trainpath=$datadir/seg.train.pd
testpath=$datadir/seg.test.pd

trainpath=$datadir/seg.part1000.pd
testpath=$datadir/seg.test1

uni_embedpath=$datadir/gigaword_chn.all.a2b.uni.ite50.vec
bi_embedpath=$datadir/gigaword_chn.all.a2b.bi.ite50.vec

uni_vocabpath=$datadir/gigaword_chn.all.a2b.uni.vocab
bi_vocabpath=$datadir/gigaword_chn.all.a2b.bi.vocab

FT_dir=$datadir/fine-tuned
mkdir -p $FT_dir
out_embedpath1=$FT_dir/gigaword_chn.all.a2b.uni.FT.vec
out_embedpath2=$FT_dir/gigaword_chn.all.a2b.bi.FT.vec

num_epochs=1
checkpoint=./cp_dir/
mkdir -p $checkpoint
modelpath=$checkpoint/seg.iter${num_epochs}.model

# modelpath=$checkpoint/seg.iter${num_epochs}.model

python train_emb.py  --num_epochs ${num_epochs} --embed_dim 50 --batch_size 10 --context_size 2 \
		--hidden_units 150 --drop_rate 0.2 --train $trainpath --test $testpath \
		--uni_vocab $uni_vocabpath --bi_vocab $bi_vocabpath --uni_embed $uni_embedpath \
		--bi_embed $bi_embedpath --modelpath $modelpath --out_embed1 $out_embedpath1 \
		--out_embed2 $out_embedpath2 --weight_dir $FT_dir --train_flag #--load_biEmb
