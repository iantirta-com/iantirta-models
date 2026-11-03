
import logging

import numpy as np
import torch

from iantirta.models import TextTiming, WordTiming

from .modeling_asr import ASRModel


class Wav2Vec2(ASRModel):
    """ Meta Facebook ASR Model Class. """
    def _load_model(self, model_name_or_path: str, **kwargs) -> None:
        from transformers import AutoModelForCTC, AutoProcessor

        merged_kwargs, kwargs = merge_kwargs(Wav2Vec2Kwargs, kwargs)
        self.config = Wav2Vec2Kwargs(merged_kwargs)

        self.model = AutoModelForCTC.from_pretrained(model_name_or_path, **self.config["common_kwargs"]).to(env.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, **self.config["common_kwargs"])
        
        if kwargs:
            logger.warning(f"Unused kwargs in {type(self).__name__}: {kwargs}")

    # ------------------
    # Helper Method
    # ------------------

    def populate_timestamp(
        self,
        audio_lists: list[np.ndarray],
        align_results: list[tuple[int, list]],
        result: list[TextTiming]
    ) -> list[TextTiming]:

        for i, (num_frames, word_spans) in enumerate(align_results):
            # _ = len(audio_lists[i]) / num_frames / self.sr
            # MMS FA Uses 20ms, todo find a better way to get it. since the stride ratio is not correct yet
            parse_timestamp = lambda t: t * 0.02

            assert len(word_spans) == len(result[i].words), (
                f"WordSpan Length: {len(word_spans)}\n"
                f"Actual Words Length: {len(result[i].words)}\n"
                f"WordSpan: {[''.join(self.processor.decode(span.token)) for spans in word_spans for span in spans]}\n"
                f"Actual Words: {[w.word for w in result[i].words]}\n"
            )

            for spans, word in zip(word_spans, result[i].words):
                word.start=parse_timestamp(spans[0].start)
                word.end=parse_timestamp(spans[-1].end)
                word.score=self.make_score(spans)

        return result

    def make_score(self, spans) -> float:
        return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)
        
    # ------------------
    # Inference Method
    # ------------------

    def inputs(self, wavs: list[np.ndarray]):
        return self.processor(
            audio=wavs,
            sampling_rate=self.sr,
            **self.config["processor_kwargs"]
        ).to(device=self.model.device, dtype=self.model.dtype)

    @torch.no_grad()
    def _infer(self, inputs):
        return self.model(**inputs)

    @torch.inference_mode()
    def _transcribe(
        self,
        wavs: list[np.ndarray],
        languages: list[str | None],
        references: list[str | None],
        *,
        return_timestamps: bool = True,
    ) -> list[TextTiming]:
        """ Model Transcribe Inference """
        audio_groups: dict[str | None, list[tuple[int, np.ndarray]]] = self.group_by_language(
            wavs, languages
        )

        results = [None] * sum(len(items) for items in audio_groups.values())

        for lang, items in audio_groups.items():
            indices = []
            sub_wavs = []
            for i, audio in items:
                indices.append(i)
                sub_wavs.append(audio)

            self.processor.tokenizer.set_target_lang(lang)
            self.model.load_adapter(lang)

            inputs = self.inputs(sub_wavs)
            outputs = self._infer(inputs)

            logits = outputs.logits
            generated_token_ids = logits.argmax(dim=-1)

            transcriptions: list[str] = self.processor.batch_decode(generated_token_ids)

            result = [
                TextTiming(
                    words=[
                        WordTiming(start=None, end=None, score=None, word=word)
                        for word in transcript.split()
                    ],
                    language=lang
                )
                for transcript in transcriptions
            ]
            if return_timestamps:

                emissions = logits.log_softmax(dim=-1)
                align_results = self._align_emissions(emissions=emissions, transcripts=transcriptions)
                result = self.populate_timestamp(sub_wavs, align_results, result)

            assert len(result) == len(indices)

            for i, res in zip(indices, result):
                results[i] = res

        # logger.debug(asr_results)
        assert len(results) == len(languages), f"transcript result length missmatch {len(results)} == {len(languages)}"

        return results

    @torch.inference_mode()
    def _align_emissions(
        self,
        emissions: torch.Tensor,
        transcripts: list[str] | None = None,
        *,
        logger: logging.Logger | None = None
    ) -> list:
        """
        Method that is used to compute alignment
        """
        import torchaudio.functional as F

        assert emissions.size(0) == len(transcripts), (
            f"Emission and ref length missmatch, {emissions.size(0)} == {len(transcripts)}"
        )

        norm_func = str.lower if "mms" in self.model.name_or_path else str.upper
        norm_refs = [norm_func(ref) for ref in transcripts]

        # `torchaudio.functional` alignment support batch == 1 only.
        assert len(norm_refs) == emissions.size(0) # Size 0 is [batch]

        results = []
        for i, norm_ref in enumerate(norm_refs):
            token_ids = self.processor.tokenizer(
                norm_ref.replace(" ", ""), **self.config["tokenizer_kwargs"]
            )["input_ids"].to(dtype=torch.long, device=self.model.device)

            local_emissions = emissions[i].unsqueeze(0)

            (logger or self.logger).debug(
                f"[{i}] Aligning Emissions {local_emissions.shape}\n"
                f"Target: {self.processor.batch_decode(token_ids)}"
            )
    
            alignments, scores = F.forced_align(
                local_emissions,
                token_ids,
                blank=self.processor.tokenizer.pad_token_id
            )

            scores = scores.exp()

            token_spans = F.merge_tokens(
                alignments[0],
                scores[0],
                blank=self.processor.tokenizer.pad_token_id
            )

            # Flatten
            words = norm_ref.split()
            (logger or self.logger).debug(f"{words}, {norm_ref}")
            n = [len(w) for w in words]
            assert sum(n) == len(token_spans), f"Length Missmatch: Token {len(token_spans)} | Words {sum(n)}"

            j, word_spans = 0, []
            for l in n:
                word_spans.append(token_spans[j:j+l])
                j+=l

            num_frames = local_emissions.size(1)
            results.append((num_frames, word_spans)) # word_spans == list[TokenSpan(start, end, token, score)]

        # Returning list[list[tuple[int, TokenSpan]]]
        # since we can't calculate its timestamp without enough data
        return results

    @torch.inference_mode()
    def _align(
        self,
        wavs: list[np.ndarray] | None = None,
        transcripts: list[str] | None = None,
        languages: list[str] | None = None,
        *,
        emissions: torch.Tensor | list[torch.Tensor] | None = None,
        logger: logging.Logger | None = None
    ) -> list[tuple[int, list]]:

        # If `audios` is pass group by lang
        # Else adapter already loaded
        if wavs is not None:
            audio_groups: dict[str | None, list[tuple[int, np.ndarray]]] = self.group_by_language(
                wavs, languages
            )

            results = [None] * sum(len(items) for items in audio_groups.values())

            for lang, items in audio_groups.items():
                indices, sub_wavs, sub_transcripts = [], [], []
                for i, wav in items:
                    indices.append(i)
                    sub_wavs.append(wav)
                    sub_transcripts.append(transcripts[i])

                self.processor.tokenizer.set_target_lang(lang)
                self.model.load_adapter(lang)

                inputs = self.inputs(sub_wavs)
                outputs = self._infer(inputs)

                logits = outputs.logits
                emissions = logits.log_softmax(dim=-1)

                (logger or self.logger).debug(
                    f"Alignment aligning, {lang} of audio length, {len(sub_transcripts)}"
                )
                group_results = self._align_emissions(
                    emissions=emissions,
                    transcripts=sub_transcripts
                )

                for i, global_i in enumerate(indices):
                    results[global_i] = group_results[i]

            return results

        elif emissions is not None:
            return self._align_emissions(
                emissions=emissions,
                transcripts=transcripts
            )

        raise ValueError("Either `audios` or `emissions` must be provided")
