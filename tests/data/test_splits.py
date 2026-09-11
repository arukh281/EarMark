"""Speaker namespaces and train vs dev/test speaker disjointness."""

from __future__ import annotations

import numpy as np
import pytest

from earmark.data.libricss import parse_meeting_info
from earmark.data.splits import (
    VB_TEST_SPEAKERS,
    SplitLeakError,
    check_speaker_disjoint,
    speaker_key,
    split_speaker_key,
)


def test_speaker_keys() -> None:
    assert speaker_key("libri", 3081) == "libri:3081"
    assert speaker_key("VCTK", "P232") == "vctk:p232"
    assert speaker_key("vctk", 225) == "vctk:p225"
    assert split_speaker_key("libri:3081") == ("libri", "3081")
    with pytest.raises(ValueError):
        speaker_key("bad:ns", "1")
    with pytest.raises(ValueError):
        split_speaker_key("3081")
    assert set(VB_TEST_SPEAKERS) == {"vctk:p232", "vctk:p257"}


def test_libricss_and_libritts_share_a_namespace() -> None:
    utts = parse_meeting_info("start\tend\tspeaker\tid\ttext\n0.5\t2.0\t1089\t1089-134686-0000\tHI\n")
    assert utts[0].speaker == speaker_key("libri", "1089")


def test_disjoint_lists_pass_and_leaks_are_named() -> None:
    train = {"libri:103", "libri:1034", "vctk:p225"}
    held = {"libritts_r_test": {"libri:1089", "libri:121"}, "vb_test": set(VB_TEST_SPEAKERS)}
    check_speaker_disjoint(train, held)
    with pytest.raises(SplitLeakError, match=r"vb_test: vctk:p232"):
        check_speaker_disjoint(train | {"vctk:p232"}, held)
    with pytest.raises(SplitLeakError, match="libritts_r_test"):
        check_speaker_disjoint(train | {"libri:1089"}, held)


@pytest.mark.parametrize("seed", range(10))
def test_random_partitions_are_disjoint(seed: int) -> None:
    """Property: a partition passes; moving any one held-out speaker into train fails."""
    rng = np.random.default_rng(seed)
    ids = [f"libri:{i}" for i in rng.choice(10_000, size=300, replace=False)]
    cut = rng.integers(50, 250)
    train, dev, test = set(ids[:cut]), set(ids[cut : cut + 25]), set(ids[cut + 25 :])
    check_speaker_disjoint(train, {"dev": dev, "test": test, "vb": VB_TEST_SPEAKERS})
    leaked = sorted(test)[int(rng.integers(len(test)))]
    with pytest.raises(SplitLeakError, match=leaked):
        check_speaker_disjoint(train | {leaked}, {"dev": dev, "test": test})
