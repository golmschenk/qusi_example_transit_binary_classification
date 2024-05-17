import shutil
from pathlib import Path

import numpy as np

from qusi.experimental.application.tess import (
    download_spoc_light_curves_for_tic_ids,
    get_spoc_tic_id_list_from_mast,
    TessToiDataInterface,
    ToiColumns,
)


def main():
    print('Retrieving metadata...')
    spoc_target_tic_ids = get_spoc_tic_id_list_from_mast()
    tess_toi_data_interface = TessToiDataInterface()
    positive_tic_ids = tess_toi_data_interface.toi_dispositions[
        tess_toi_data_interface.toi_dispositions[ToiColumns.disposition.value] != 'FP'][ToiColumns.tic_id.value]
    negative_tic_ids = list(set(spoc_target_tic_ids) - set(positive_tic_ids))
    positive_tic_ids_splits = np.split(
        np.array(positive_tic_ids), [int(len(positive_tic_ids) * 0.8), int(len(positive_tic_ids) * 0.9)])
    positive_train_tic_ids = positive_tic_ids_splits[0].tolist()
    positive_validation_tic_ids = positive_tic_ids_splits[1].tolist()
    positive_test_tic_ids = positive_tic_ids_splits[2].tolist()
    negative_tic_ids_splits = np.split(
        np.array(negative_tic_ids), [int(len(negative_tic_ids) * 0.8), int(len(negative_tic_ids) * 0.9)])
    negative_train_tic_ids = negative_tic_ids_splits[0].tolist()
    negative_validation_tic_ids = negative_tic_ids_splits[1].tolist()
    negative_test_tic_ids = negative_tic_ids_splits[2].tolist()
    sectors = list(range(27, 56))

    print('Downloading light curves...')
    download_spoc_light_curves_for_tic_ids(
        tic_ids=positive_train_tic_ids,
        download_directory=Path('data/spoc_transit_experiment/train/positives'),
        sectors=sectors,
        limit=2000)
    download_spoc_light_curves_for_tic_ids(
        tic_ids=negative_train_tic_ids,
        download_directory=Path('data/spoc_transit_experiment/train/negatives'),
        sectors=sectors,
        limit=6000)
    download_spoc_light_curves_for_tic_ids(
        tic_ids=positive_validation_tic_ids,
        download_directory=Path('data/spoc_transit_experiment/validation/positives'),
        sectors=sectors,
        limit=200)
    download_spoc_light_curves_for_tic_ids(
        tic_ids=negative_validation_tic_ids,
        download_directory=Path('data/spoc_transit_experiment/validation/negatives'),
        sectors=sectors,
        limit=600)
    download_spoc_light_curves_for_tic_ids(
        tic_ids=positive_test_tic_ids,
        download_directory=Path('data/spoc_transit_experiment/test/positives'),
        sectors=sectors,
        limit=200)
    download_spoc_light_curves_for_tic_ids(
        tic_ids=negative_test_tic_ids,
        download_directory=Path('data/spoc_transit_experiment/test/negatives'),
        sectors=sectors,
        limit=600)
    # In this toy example, we reuse our test light curves as infer light curves. In a real world case, you would likely
    #  want to infer on cases you don't already know the answer for.
    infer_directory_path = Path('data/spoc_transit_experiment/infer')
    infer_directory_path.mkdir(exist_ok=True, parents=True)
    for light_curve_path in Path('data/spoc_transit_experiment/test').glob('**/*.fits'):
        shutil.copy(light_curve_path, infer_directory_path.joinpath(light_curve_path.name))


if __name__ == '__main__':
    main()
