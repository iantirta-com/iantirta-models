
from dataclasses import dataclass

from iantirta.audio import HumanTiming, TimingMixin
import iantirta.text as text_utils

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

    language: str | None = None

    _clean: str | None = None

    @property
    def clean(self) -> str:
        if self._clean is None:
            self._clean = text_utils.normalize(self.word, kaldi=True)
        return self._clean


@dataclass(slots=True)
class TextTiming(HumanTiming):
    """
    Class that represent ASR/FA Text timestamp timing

    :param list[WordTiming] words:
    """
    words: list[WordTiming]
    language: str | None = None
    ass_event: str | None = None

    _clean: str | None = None
    _text: str | None = None

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

    @property
    def clean(self) -> str:
        if self._clean is None:
            self._clean = " ".join([w.clean for w in self.words])
        return self._clean

    @property
    def text(self) -> str:
        if self._text is None:
            self._text = " ".join([w.word for w in self.words])
        return self._text