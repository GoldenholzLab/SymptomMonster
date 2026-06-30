"""Turn one patient's notes into a list of model-attributed symptoms.

The prompt names the index treatment by its masked token, and the masker rewrites
that same drug to that same token inside the notes, so the model reads a single
consistent fictitious name and never sees the real one. Masking happens before
the text leaves this module, and a leak check guards the boundary.
"""

from __future__ import annotations

import time
from importlib.resources import files

from symptommonster.io import ExtractionRecord, PatientNotes
from symptommonster.llm import LLMClient, LLMRequest, extract_string_list
from symptommonster.masking import DrugMasker

_WINDOW_SEPARATOR = "\n\n---\n\n"

# Used when a patient carries no cohort label and the index drug is therefore
# implicit. Neutral on purpose: it must not leak any real drug identity.
_DEFAULT_INDEX_PHRASE = "the index medication"

# Truncation guard for one note window, sized to the model context: about 131,072
# tokens at roughly four characters per token. Longer windows are clipped before
# masking so a pathological note set cannot run past the context.
_MAX_WINDOW_CHARS = 524_288


class SymptomExtractor:
    """Extract symptoms attributable to a patient's index treatment.

    `alias_map` maps every real drug name to its masking token and doubles as the
    group-to-index lookup: a patient's `group` is expected to be the real index
    drug name, whose token is what the prompt refers to.
    """

    def __init__(
        self,
        client: LLMClient,
        alias_map: dict[str, str],
        template: str,
        *,
        temperature: float = 0.05,
        max_tokens: int = 4096,
        max_chars: int = _MAX_WINDOW_CHARS,
    ) -> None:
        self.client = client
        self.alias_map = dict(alias_map)
        self.template = template
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.masker = DrugMasker(alias_map)

    def _prepare_window(self, notes: list[str]) -> str:
        """Join, truncate, and mask one note window."""
        joined = _WINDOW_SEPARATOR.join(notes)[: self.max_chars]
        masked = self.masker.mask(joined)
        # The notes are the only untrusted text reaching the model; fail loudly
        # rather than ship a window that still names a real drug.
        self.masker.assert_no_leak(masked)
        return masked

    def _index_token(self, group: str | None) -> str:
        if group is None:
            return _DEFAULT_INDEX_PHRASE
        token = self.alias_map.get(group)
        if token is None:
            raise ValueError(
                f"cohort {group!r} has no entry in the alias map; include the "
                "cohort's index drug so it can be masked and named in the prompt"
            )
        return token

    def extract(self, notes: PatientNotes) -> ExtractionRecord:
        pre = self._prepare_window(notes.pre_notes)
        post = self._prepare_window(notes.post_notes)
        prompt = render(
            self.template,
            alias=self._index_token(notes.group),
            pre_notes=pre,
            post_notes=post,
        )
        request = LLMRequest(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        start = time.perf_counter()
        response = self.client.generate(request)
        elapsed = time.perf_counter() - start

        symptoms = extract_string_list(response, key="side_effects")
        return ExtractionRecord(
            patient_id=notes.patient_id,
            group=notes.group,
            symptoms=symptoms,
            raw_response=response,
            extraction_time_s=elapsed,
        )


def load_template(path: str | None = None) -> str:
    """Return the prompt template, from `path` if given, else the packaged one."""
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    resource = files("symptommonster.resources.prompts") / "extraction.txt"
    return resource.read_text(encoding="utf-8")


def render(template: str, *, alias: str, pre_notes: str, post_notes: str) -> str:
    """Fill the {alias}, {pre_notes}, and {post_notes} slots.

    Slots are filled by literal replacement rather than str.format, because
    clinical note text routinely contains stray braces that would break it.
    """
    return (
        template.replace("{alias}", alias)
        .replace("{pre_notes}", pre_notes)
        .replace("{post_notes}", post_notes)
    )
