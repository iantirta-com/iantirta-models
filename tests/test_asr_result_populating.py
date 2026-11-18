
import pytest

from iantirta.models import ASRResult


def test_populate_ass(final_res: ASRResult):
    final_res.populate_ass()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([
        '-v', '-s',
        '--log-cli-level=DEBUG',
        __file__
    ]))
