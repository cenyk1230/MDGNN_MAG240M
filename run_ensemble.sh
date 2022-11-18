
MAG_CODE_PATH=./ # The code path
MAG_BASE_PATH=./
MAG_INPUT_PATH=$MAG_BASE_PATH/dataset_path/ # The MAG240M-LSC dataset should be placed here
MAG_PREP_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/preprocess/
MAG_RGAT_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/rgat/
MAG_FEAT_PATH=$MAG_BASE_PATH/dataset_path/mplp_data/feature/


MAG_SUBM_PATH=$MAG_INPUT_PATH/mplp_data/subm
MAG_METHODS_PATH=$MAG_INPUT_PATH/mplp_data/outputs

mkdir -p $MAG_SUBM_PATH
python3 $MAG_CODE_PATH/ensemble_last.py $MAG_INPUT_PATH $MAG_METHODS_PATH $MAG_SUBM_PATH $MAG_FEAT_PATH