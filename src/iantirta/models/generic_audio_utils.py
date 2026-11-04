
from dataclasses import dataclass

from iantirta.audio import HumanTiming, TimingMixin

__all__ = [
    "TextTiming",
    "WordTiming",
]


@dataclass(slots=True)
class WordTiming(TimingMixin):
    """
    Class that represent ASR/FA Word timestamp timing

    :param str word:
    :param float score:
        the confidence score for this word.
    """
    word: str
    score: float


@dataclass(slots=True)
class TextTiming(HumanTiming):
    """
    Class that represent ASR/FA Text timestamp timing

    :param list[WordTiming] words:
    """
    words: list[WordTiming]


    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def duration(self) -> float:
        if self.start is None or self.end is None: return 0.0
        return float(round(self.end - self.start, 2))
