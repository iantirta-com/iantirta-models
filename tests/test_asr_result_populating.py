
import pytest

from iantirta.models import ASRResult


def test_populate_ass(fa_res: ASRResult, indo_lyric: str):
    fa_res.to_line_idx(indo_lyric)
    fa_res.populate_ass()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([
        '-v', '-s',
        '--log-cli-level=DEBUG',
        __file__
    ]))
