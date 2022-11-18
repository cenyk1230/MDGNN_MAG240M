#!/usr/bin/env bash
export DGLBACKEND=pytorch

MAG_CODE_PATH=./ # The code path
MAG_BASE_PATH=./
MAG_INPUT_PATH=$MAG_BASE_PATH/dataset_path/ # The MAG240M-LSC dataset should be placed here
MAG_PREP_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/preprocess/
MAG_RGAT_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/rgat/
MAG_FEAT_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/feature/

# preprocess grph data
mkdir -p $MAG_PREP_PATH
python3 $MAG_CODE_PATH/preprocess.py
        --rootdir $MAG_INPUT_PATH \
        --author-output-path $MAG_PREP_PATH/author.npy \
        --inst-output-path $MAG_PREP_PATH/inst.npy \
        --graph-output-path $MAG_PREP_PATH \
        --graph-as-homogeneous \
        --full-output-path $MAG_PREP_PATH/full_feat.npy

# generate meta-path based features
mkdir -p $MAG_FEAT_PATH
python3 $MAG_CODE_PATH/features/feature.py
        $MAG_INPUT_PATH \
        $MAG_PREP_PATH/dgl_graph_full_heterogeneous_csr.bin \
        $MAG_FEAT_PATH \
        $MAG_PREP_PATH \
        --seed=42

# run rgats (baseline)
mkdir -p $MAG_RGAT_PATH
python3 $MAG_CODE_PATH/features/rgat.py
        --rootdir $MAG_INPUT_PATH \
        --graph-path $MAG_PREP_PATH/dgl_graph_full_homogeneous_csc.bin \
        --full-feature-path $MAG_PREP_PATH/full_feat.npy \
        --output-path $MAG_RGAT_PATH/ \
        --epochs=100 \
        --model-path $MAG_RGAT_PATH/model.pt \

# run rgats (three layers)
python $MAG_CODE_PATH/features/rgat_3layers.py \
        --rootdir $MAG_INPUT_PATH \
        --graph-path $MAG_PREP_PATH/dgl_graph_full_homogeneous_csc.bin \
        --full-feature-path $MAG_PREP_PATH/full_feat.npy \
        --output-path $MAG_RGAT_PATH/ \
        --epochs=100 \
        --model-path $MAG_RGAT_PATH/3layer_model.pt \

# copy rgat embeddings
cp -n $MAG_RGAT_PATH/*x_rgat_*.npy $MAG_FEAT_PATH

