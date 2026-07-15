import sys, os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from judges.local_hf import get_judge


# Only the dispatch logic is unit-tested here; constructing a real judge downloads 7B models
# and needs a GPU, so that is smoke-tested on the cluster (plan Task 4, Step 2).
def test_get_judge_unknown_concept_raises():
    with pytest.raises(ValueError):
        get_judge("mystery-concept")
