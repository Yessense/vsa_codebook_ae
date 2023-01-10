mkdir -p ./data/paired_dsprites/
wget https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true -O "./data/paired_dsprites/dsprites_train.npz"
#python3 -m vsa_codebook_decoder.dataset.paired_dsprites --mode make_dataset --max_exchanges 1\
#  --train_size 100000 --test_size 30000\
#  --path_to_dsprites_train "./data/paired_dsprites/dsprites_train.npz"\
#  --save_path "./data/paired_dsprites/"
