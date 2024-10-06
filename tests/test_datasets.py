import numpy as np
import pytest
from braindecode.preprocessing import Preprocessor, preprocess

from eegatscale.datasets import TUH


@pytest.fixture(scope="module")
def offline():
    return TUH(path="/nobackup/tsal-tmp/tuh_eeg/", recording_ids=list(range(25)), offline=True)


@pytest.fixture(scope="module")
def online():
    return TUH(path="/nobackup/tsal-tmp/tuh_eeg/", recording_ids=list(range(25)), offline=False)


def test_dataset_attribute(offline, online):
    # len matches
    assert len(offline.datasets) == len(online.datasets)

    # index works the same way
    assert all(offline.datasets[0].description == online.datasets[0].description)
    assert offline.datasets[0].raw.load_data() == online.datasets[0].raw.load_data()

    # iterating works
    for off, on in zip(offline.datasets, online.datasets, strict=True):
        assert all(off.description == on.description)
        assert off.raw.load_data() == on.raw.load_data()
        assert np.array_equal(off.raw.get_data(), on.raw.get_data())


def test_with_preprocess(offline, online):
    resample = Preprocessor("resample", sfreq=10)
    preprocessed_offline = preprocess(offline, preprocessors=[resample])

    # should be given as a argument during initialization, but patch for testing purpose
    online._preprocess = [resample]

    # check index works with preprocessing
    assert np.array_equal(preprocessed_offline.datasets[0].raw.get_data(), online.datasets[0].raw.get_data())

    # check that iteration works with processing
    for off, on in zip(preprocessed_offline.datasets, online.datasets):
        assert all(off.description == on.description)
        assert off.raw.load_data() == on.raw.load_data()
        assert np.array_equal(off.raw.get_data(), on.raw.get_data())
