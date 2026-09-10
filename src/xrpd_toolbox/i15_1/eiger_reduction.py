from collections.abc import Callable
from pathlib import Path

from xrpd_toolbox.i15_1.eiger_500k import EigerDataLoader
from xrpd_toolbox.utils.utils import wait_for_finished_file


def get_eiger_mask(pixel_mask_filepath: str):

    mask_filepath, mask_datapath = pixel_mask_filepath.split("//")
    eiger_data = EigerDataLoader(mask_filepath)
    return eiger_data.get_data(mask_datapath)


def do_eiger_data_reduction(nexus: str | Path):
    eiger_data = EigerDataLoader(nexus)

    print(eiger_data.positions)

    pixel_mask_filepath = eiger_data.get_pixel_mask_path()
    mask = get_eiger_mask(pixel_mask_filepath)

    print(mask)

    print(pixel_mask_filepath)

    print(eiger_data.is_background())

    print(eiger_data.get_plan_name())


collection_analysis_dict: dict[str, Callable] = {
    "data_collection": do_eiger_data_reduction,
}


def run_eiger_analysis(nexus: str | Path):

    wait_for_finished_file(nexus)

    eiger_data = EigerDataLoader(nexus)
    plan_name = eiger_data.get_plan_name()

    print(plan_name)

    plan_to_run = collection_analysis_dict[plan_name]

    if plan_to_run is not None:
        plan_to_run(nexus)


if __name__ == "__main__":
    nexus = "/workspaces/outputs/i15-1/i15-1-98478.nxs"
    run_eiger_analysis(nexus)
