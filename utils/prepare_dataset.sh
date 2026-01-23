#!/bin/bash

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATASET_DIR="${1}"
CAMERA_MODEL="${2:kb4}"
PATTERN_SIZE="${3:small}"

function usage()
{
    echo "Usage: ./$( basename $0 ) <path_to_dataset> [camera_model] [pattern_size]"
    exit 1
}

if [ "$#" -ne 1 ] && [ "$#" -ne 2 ] && [ "$#" -ne 3 ]; then
    echo 'Invalid number of arguments. Provide the path to the dataset only.'
    usage
fi

if [ ! -d "${DATASET_DIR}" ]; then
    echo "'${DATASET_DIR}' is not a valid directory. Check the provided path."
    usage
fi

if [ ! -d "${DATASET_DIR}/Calibrations" ]; then
    echo "Dataset not found at '${DATASET_DIR}'. Check the provided path."
    usage
fi

# copy required files to calibration folder
cp ${SCRIPT_DIR}/vicalib_calibration_and_poses.sh ${DATASET_DIR}/Calibrations || exit 2
cp ${SCRIPT_DIR}/pattern_${PATTERN_SIZE}.xml ${DATASET_DIR}/Calibrations || exit 2
cp ${SCRIPT_DIR}/mask.png ${DATASET_DIR}/Calibrations || exit 2

cd ${DATASET_DIR}/Calibrations
# copy vicalib pattern's properties to each endoscope folder
for i in c*_*_video; do cp pattern_${PATTERN_SIZE}.xml "${i}/${i}_pattern.xml" || exit 2; done
# copy 2D boolean mask to each endoscope folder
for i in c*_*_video; do cp mask.png "${i}/${i}_mask.png" || exit 2; done
# extract frames from video and get poses from previous geometrical calibration
for i in c*_*_video; do ./vicalib_calibration_and_poses.sh ${i} ${PATTERN_SIZE} ${CAMERA_MODEL} || exit 2;  done

# clean temporal files
rm ${DATASET_DIR}/Calibrations/vicalib_calibration_and_poses.sh
rm ${DATASET_DIR}/Calibrations/pattern_${PATTERN_SIZE}.xml
rm ${DATASET_DIR}/Calibrations/mask.png

cd ${CURRENT_DIR}