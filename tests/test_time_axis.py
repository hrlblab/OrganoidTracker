from src.analysis.organoid_cyst_data import ExperimentData


def test_days_are_numbered_from_one_as_in_the_paper():
    experiment = ExperimentData(total_frames=7, time_lapse_days=6.0, conversion_factor_um_per_pixel=1.0)
    assert experiment.frame_timestamps == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert experiment.get_time_at_frame(0) == 1.0 and experiment.get_time_at_frame(6) == 7.0


def test_single_frame_and_uneven_span():
    assert ExperimentData(total_frames=1, time_lapse_days=6.0, conversion_factor_um_per_pixel=1.0).frame_timestamps == [1.0]
    experiment = ExperimentData(total_frames=4, time_lapse_days=1.5, conversion_factor_um_per_pixel=1.0)
    assert experiment.frame_timestamps == [1.0, 1.5, 2.0, 2.5]
