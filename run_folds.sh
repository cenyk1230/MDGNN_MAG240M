#!/usr/bin/env bash
export DGLBACKEND=pytorch
# export CUDA_VISIBLE_DEVICES=0

MAG_CODE_PATH=./ # The code path
MAG_BASE_PATH=./
MAG_INPUT_PATH=$MAG_BASE_PATH/dataset_path/ # The MAG240M-LSC dataset should be placed here
MAG_PREP_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/preprocess/
MAG_RGAT_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/rgat/
MAG_FEAT_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/feature/

# for base parameter model
MODELS=(
    "rgat1024_label" 
    "rgat1024_label_m2v_feat_3rgat1024" 
    "label_m2v_3rgat1024" 
    "rgat1024_label_m2v_feat_3rgat1024_jax_fhid" 
)
for MODEL_INDEX in {0..4}; 
do
    echo ${MODELS[${MODEL_INDEX}]}
    MAG_OUTPUT_PATH=$MAG_INPUT_PATH/mplp_data/outputs/${MODELS[${MODEL_INDEX}]}
    # echo $MAG_OUTPUT_PATH
    mkdir -p $MAG_OUTPUT_PATH
    python3 $MAG_CODE_PATH/mplp_folds.py \
            $MAG_INPUT_PATH \
            $MAG_FEAT_PATH \
            $MAG_OUTPUT_PATH \
            --gpu \
            --finetune \
            --seed=0 \
            --batch_size=10000 \
            --epochs=200 \
            --num_layers=2 \
            --learning_rate=0.01 \
            --mlp_hidden=512 \
            --dropout=0.5 \
done

# for hidden 1024 model
MODELS=(
    "rgat1024_label_m2v_feat_fhid_hid1024" 
)
for MODEL_INDEX in {0..0}; 
do
    echo ${MODELS[${MODEL_INDEX}]}
    MAG_OUTPUT_PATH=$MAG_INPUT_PATH/mplp_data/outputs/${MODELS[${MODEL_INDEX}]}
    # # echo $MAG_OUTPUT_PATH
    mkdir -p $MAG_OUTPUT_PATH
    python3 $MAG_CODE_PATH/mplp_folds.py \
            $MAG_INPUT_PATH \
            $MAG_FEAT_PATH \
            $MAG_OUTPUT_PATH \
            --gpu \
            --finetune \
            --seed=0 \
            --batch_size=10000 \
            --epochs=200 \
            --num_layers=2 \
            --learning_rate=0.01 \
            --mlp_hidden=1024 \
            --dropout=0.5 \
done

# for seeds model
MODELS=(
    "rgat1024_label_m2v_feat_fhid" 
    "rgat1024_label_m2v_feat_3rgat1024_fhid" 
)
for MODEL_INDEX in {0..1}; 
do
    echo ${MODELS[${MODEL_INDEX}]}
    if [ ${MODEL_INDEX} -eq 0 ];
    then
        SEEDS=6
    fi
    if [ ${MODEL_INDEX} -eq 1 ];
    then
        SEEDS=3
    fi
    
    n=0
    while [ -n n ];
    do
        MAG_OUTPUT_PATH=$MAG_INPUT_PATH/mplp_data/outputs/${MODELS[${MODEL_INDEX}]}_seed$n
        echo $MAG_OUTPUT_PATH
        mkdir -p $MAG_OUTPUT_PATH
        python3 $MAG_CODE_PATH/mplp_folds.py \
                $MAG_INPUT_PATH \
                $MAG_FEAT_PATH \
                $MAG_OUTPUT_PATH \
                --gpu \
                --finetune \
                --seed=n \
                --batch_size=10000 \
                --epochs=200 \
                --num_layers=2 \
                --learning_rate=0.01 \
                --mlp_hidden=512 \
                --dropout=0.5 \
        if [ $n = $SEEDS ];
        then
            break
        else
            let n++
            continue
        fi
    done
    
done

# POST_SMOOTHING_GRAPH_PATH=$MAG_PREP_PATH/ppgraph_cite.bin
# POST_SMOOTHING_GRAPH_PATH=$MAG_PREP_PATH/paper_coauthor_paper_symmetric_jc0.5.bin
# python3 $MAG_CODE_PATH/post_s.py $MAG_INPUT_PATH $MAG_OUTPUT_PATH $POST_SMOOTHING_GRAPH_PATH


MAG_SUBM_PATH=$MAG_OUTPUT_PATH/subm
MAG_METHODS_PATH=$MAG_MPLP_PATH/outputs
# mkdir -p $MAG_SUBM_PATH
python3 $MAG_CODE_PATH/ensemble_folds.py $MAG_INPUT_PATH $MAG_OUTPUT_PATH $MAG_SUBM_PATH
python3 $MAG_CODE_PATH/ensemble_last.py $MAG_INPUT_PATH $MAG_METHODS_PATH $MAG_SUBM_PATH

echo 'DONE!'
