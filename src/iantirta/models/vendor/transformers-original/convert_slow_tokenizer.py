# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities to convert slow tokenizers in their fast tokenizers counterparts.

All the conversions are grouped here to gather SentencePiece dependencies outside of the fast tokenizers files and
allow to make our dependency on SentencePiece optional.
"""

import warnings

from packaging import version
from tokenizers import (
    AddedToken,
    Regex,
    Tokenizer,
    decoders,
    normalizers,
    pre_tokenizers,
    processors,
)
from tokenizers.models import BPE, Unigram

from .utils import (
    is_protobuf_available,
    is_sentencepiece_available,
    logging,
    requires_backends,
)
from .utils.import_utils import PROTOBUF_IMPORT_ERROR

logger = logging.get_logger(__name__)


def import_protobuf(error_message=""):
    if is_sentencepiece_available():
        from sentencepiece import sentencepiece_model_pb2

        return sentencepiece_model_pb2
    if is_protobuf_available():
        import google.protobuf

        if version.parse(google.protobuf.__version__) < version.parse("4.0.0"):
            from iantirta.models.vendor.transformers.utils import sentencepiece_model_pb2
        else:
            from iantirta.models.vendor.transformers.utils import sentencepiece_model_pb2_new as sentencepiece_model_pb2
        return sentencepiece_model_pb2
    else:
        raise ImportError(PROTOBUF_IMPORT_ERROR.format(error_message))


def _get_prepend_scheme(add_prefix_space: bool, original_tokenizer) -> str:
    if add_prefix_space:
        prepend_scheme = "always"
        if not getattr(original_tokenizer, "legacy", True):
            prepend_scheme = "first"
    else:
        prepend_scheme = "never"
    return prepend_scheme


class SentencePieceExtractor:
    """
    Extractor implementation for SentencePiece trained models. https://github.com/google/sentencepiece
    """

    def __init__(self, model: str):
        requires_backends(self, "sentencepiece")
        requires_backends(self, "protobuf")

        # from .utils import sentencepiece_model_pb2 as model_pb2
        model_pb2 = import_protobuf()

        m = model_pb2.ModelProto()
        with open(model, "rb") as f:
            m.ParseFromString(f.read())
        self.proto = m

    def extract(self, model_type, **kwargs) -> tuple[dict[str, int], list[tuple]]:
        """
        By default will return vocab and merges with respect to their order, by sending `vocab_scores` we're going to
        order the merges with respect to the piece scores instead.
        """
        self.proto.trainer_spec.unk_id
        if model_type is None:
            from tokenizers.models import BPE, Unigram

            model_type = Unigram if self.proto.trainer_spec.model_type == 1 else BPE
        vocab = [(piece.piece, piece.score) for piece in self.proto.pieces]

        if model_type.__name__ != "BPE":
            kwargs["unk_id"] = self.proto.trainer_spec.unk_id
            kwargs["vocab"] = vocab
        else:
            from .tokenization_utils_base import generate_merges

            vocab = {word: i for i, (word, score) in enumerate(vocab)}
            merges = generate_merges(vocab)
            kwargs["vocab"] = vocab
            kwargs["merges"] = merges

        # control tokens are special
        # user defined symbols are not
        # both user and control tokens are AddedTokens
        # Add user defined symbols (type == 4) from sentencepiece (https://github.com/google/sentencepiece/blob/6225e08edb2577757163b3f5dbba4c0b670ef445/src/sentencepiece_model.proto#L299C29-L299C33)
        spm_added_tokens = [(id, p.piece, p.type == 3) for id, p in enumerate(self.proto.pieces) if p.type in [3, 4]]
        kwargs["additional_special_tokens"] = [
            AddedToken(token, normalized=False, special=special)
            for id, token, special in sorted(spm_added_tokens, key=lambda x: x[0])
        ]
        kwargs["_spm_precompiled_charsmap"] = getattr(self.proto.normalizer_spec, "precompiled_charsmap", None)
        return kwargs


class Converter:
    def __init__(self, original_tokenizer):
        self.original_tokenizer = original_tokenizer

    def converted(self) -> Tokenizer:
        raise NotImplementedError()


class SpmConverter(Converter):
    handle_byte_fallback = False
    SpmExtractor = SentencePieceExtractor
    special_tokens = {}

    @staticmethod
    def build_tokenizer_from_spm_proto(proto, vocab, merges=None):
        """
        Similar to convert_from_spm method, but used only when there is no `model_type` class, i.e. there is no matching class in `TOKENIZERS_MAPPING` and we just create a tokenizer instead of extracting stuff from the sentencepiece file
        """
        byte_fallback = proto.trainer_spec.byte_fallback
        unk_piece = proto.trainer_spec.unk_piece
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap

        # model
        if isinstance(vocab, dict):
            tokenizer = Tokenizer(
                BPE(
                    vocab=vocab,
                    merges=merges or [],
                    unk_token=unk_piece,
                    fuse_unk=True,
                    byte_fallback=byte_fallback,
                    dropout=None,
                )
            )
        elif isinstance(vocab, list) and vocab and isinstance(vocab[0], tuple | list):
            tokenizer = Tokenizer(
                Unigram(
                    vocab=vocab,
                    unk_id=proto.trainer_spec.unk_id,
                    byte_fallback=byte_fallback,
                )
            )
        else:
            return None

        # normalizer
        _normalizers = [normalizers.Replace(" ", "▁")]
        if precompiled_charsmap:
            _normalizers.insert(0, normalizers.Precompiled(precompiled_charsmap))
        tokenizer.normalizer = normalizers.Sequence(_normalizers)

        # decoder
        if byte_fallback:
            tokenizer.decoder = decoders.Sequence(
                [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()]
            )
        else:
            tokenizer.decoder = decoders.Sequence([decoders.Replace("▁", " ")])

        return tokenizer

    @classmethod
    def convert_from_spm(cls, vocab=None, **kwargs):
        """
        Hook used when converting directly from a SentencePiece model without a slow tokenizer instance.
        By default, return kwargs unchanged.
        """
        if vocab is not None:
            kwargs["vocab"] = vocab
        return kwargs

    def __init__(self, *args):
        requires_backends(self, "protobuf")

        super().__init__(*args)

        # from .utils import sentencepiece_model_pb2 as model_pb2
        model_pb2 = import_protobuf()

        m = model_pb2.ModelProto()
        with open(self.original_tokenizer.vocab_file, "rb") as f:
            m.ParseFromString(f.read())
        self.proto = m

        if self.proto.trainer_spec.byte_fallback and not self.handle_byte_fallback:
            warnings.warn(
                "The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option"
                " which is not implemented in the fast tokenizers. In practice this means that the fast version of the"
                " tokenizer can produce unknown tokens whereas the sentencepiece version would have converted these "
                "unknown tokens into a sequence of byte tokens matching the original piece of text."
            )

    def vocab(self, proto):
        return [(piece.piece, piece.score) for piece in proto.pieces]

    def unk_id(self, proto):
        return proto.trainer_spec.unk_id

    def tokenizer(self, proto):
        model_type = proto.trainer_spec.model_type
        vocab_scores = self.vocab(proto)

        if model_type == 1:
            tokenizer = Tokenizer(
                Unigram(
                    vocab_scores,
                    unk_id=self.unk_id(proto),
                    byte_fallback=self.handle_byte_fallback,
                )
            )

        elif model_type == 2:
            _, merges = self.SpmExtractor(self.original_tokenizer.vocab_file).extract(vocab_scores)
            bpe_vocab = {word: i for i, (word, score) in enumerate(vocab_scores)}
            tokenizer = Tokenizer(
                BPE(
                    bpe_vocab,
                    merges,
                    unk_token=proto.trainer_spec.unk_piece,
                    fuse_unk=True,
                    byte_fallback=self.handle_byte_fallback,
                    dropout=None,
                )
            )

        else:
            raise Exception(
                "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )

        # control tokens are special
        # user defined symbols are not
        # both user and control tokens are AddedTokens
        # Add user defined symbols (type == 4) from sentencepiece (https://github.com/google/sentencepiece/blob/6225e08edb2577757163b3f5dbba4c0b670ef445/src/sentencepiece_model.proto#L299C29-L299C33)
        spm_added_tokens = [
            (id, p.piece, p.type == 3 or p.piece in self.special_tokens)
            for id, p in enumerate(proto.pieces)
            if p.type in [3, 4]
        ]
        tokenizer.add_tokens(
            [
                AddedToken(token, normalized=False, special=special)
                for id, token, special in sorted(spm_added_tokens, key=lambda x: x[0])
            ]
        )

        return tokenizer

    def normalizer(self, proto):
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        _normalizers = [
            normalizers.Strip(left=False, right=True),  # stripping is important
            normalizers.Replace(Regex(" {2,}"), "▁"),
        ]
        if not precompiled_charsmap:
            return normalizers.Sequence(_normalizers)
        else:
            return normalizers.Sequence([normalizers.Precompiled(precompiled_charsmap)] + _normalizers)

    def pre_tokenizer(self, replacement, add_prefix_space):
        prepend_scheme = _get_prepend_scheme(add_prefix_space, self.original_tokenizer)
        return pre_tokenizers.Metaspace(replacement=replacement, prepend_scheme=prepend_scheme)

    def post_processor(self):
        return None

    def decoder(self, replacement, add_prefix_space):
        prepend_scheme = _get_prepend_scheme(add_prefix_space, self.original_tokenizer)
        return decoders.Metaspace(replacement=replacement, prepend_scheme=prepend_scheme)

    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer(self.proto)

        # Tokenizer assemble
        normalizer = self.normalizer(self.proto)
        if normalizer is not None:
            tokenizer.normalizer = normalizer

        replacement = "▁"
        add_prefix_space = True
        if hasattr(self.original_tokenizer, "add_prefix_space"):
            add_prefix_space = self.original_tokenizer.add_prefix_space

        pre_tokenizer = self.pre_tokenizer(replacement, add_prefix_space)
        if pre_tokenizer is not None:
            tokenizer.pre_tokenizer = pre_tokenizer

        tokenizer.decoder = self.decoder(replacement, add_prefix_space)
        post_processor = self.post_processor()
        if post_processor:
            tokenizer.post_processor = post_processor

        return tokenizer


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class TikTokenConverter:
    """
    A general tiktoken converter.
    """

    def __init__(
        self,
        vocab_file=None,
        pattern=r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
        add_prefix_space=False,
        extra_special_tokens=None,
        **kwargs,
    ):
        self.vocab_file = vocab_file
        self.pattern = pattern
        self.add_prefix_space = add_prefix_space
        self.extra_special_tokens = (
            extra_special_tokens.keys() if isinstance(extra_special_tokens, dict) else extra_special_tokens
        )

    @staticmethod
    def load_tiktoken_bpe(tiktoken_url: str) -> dict[bytes, int]:
        try:
            from tiktoken.load import load_tiktoken_bpe
        except Exception:
            raise ValueError(
                "`tiktoken` is required to read a `tiktoken` file. Install it with `pip install tiktoken`."
            )
        return load_tiktoken_bpe(tiktoken_url)

    def extract_vocab_merges_from_model(self, tiktoken_url: str):
        bpe_ranks = self.load_tiktoken_bpe(tiktoken_url)
        byte_encoder = bytes_to_unicode()

        def token_bytes_to_string(b):
            return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

        merges = []
        vocab = {}
        for token, rank in bpe_ranks.items():
            vocab[token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            local = []
            for index in range(1, len(token)):
                piece_l, piece_r = token[:index], token[index:]
                if piece_l in bpe_ranks and piece_r in bpe_ranks and (piece_l + piece_r) in bpe_ranks:
                    local.append((piece_l, piece_r, rank))
            local = sorted(local, key=lambda x: (bpe_ranks[x[0]], bpe_ranks[x[1]]), reverse=False)
            merges.extend(local)
        merges = sorted(merges, key=lambda val: val[2], reverse=False)
        merges = [(token_bytes_to_string(val[0]), token_bytes_to_string(val[1])) for val in merges]
        return vocab, merges

    def tokenizer(self):
        vocab_scores, merges = self.extract_vocab_merges_from_model(self.vocab_file)
        tokenizer = Tokenizer(BPE(vocab_scores, merges, fuse_unk=False))
        if hasattr(tokenizer.model, "ignore_merges"):
            tokenizer.model.ignore_merges = True
        return tokenizer

    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer()
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(self.pattern), behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=self.add_prefix_space, use_regex=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()

        if self.extra_special_tokens is not None:
            tokenizer.add_special_tokens(
                [AddedToken(token, normalized=False, special=True) for token in self.extra_special_tokens]
            )

        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        return tokenizer


SLOW_TO_FAST_CONVERTERS = {}

