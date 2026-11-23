"""Viral-clip discovery and review via DeepSeek.

Flow (per the build spec):
1. **Discovery** — the Discovery prompt (Settings > Viral Clips, defaults
   from ``settings.DEFAULT_DISCOVERY_PROMPT``) plus a clip-structure block
   requiring 40–45 s self-contained clips with Hook (0–3 s), Lock-in zone
   (3–7 s), Body Block 1, Rehook 1, Rehook 2, Body Block 3, optional
   Closing, CTA. Thinking mode enabled at high effort (research §2.5).
2. **Review** — the Review prompt judges each proposed clip and keeps the
   strongest up to the target count, timestamps unchanged.
3. **Caption** — per finished clip, DeepSeek writes a TikTok caption and
   EXACTLY 4 relevant hashtags in the clip's language; thinking disabled
   for speed and cost.

The model returns JSON; parsing is tolerant (markdown fences, ``clips`` /
``moments`` / bare-list shapes) but never invents timestamps — every
returned range is validated (end > start, clamped to the video duration)
and dropped when it is not a real range from the transcript.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..providers._http import ProviderError
from ..providers.deepseek import DeepSeekClient
from ..settings import DEFAULT_DISCOVERY_PROMPT, DEFAULT_REVIEW_PROMPT
from .transcription import Transcript, Word

#: Two clips count as overlapping when they share more than this many
#: seconds — a tiny touch at a boundary (rounding) is not an overlap.
OVERLAP_TOLERANCE = 0.5


def dedupe_overlapping(
    candidates: list[ClipCandidate], *, tolerance: float = OVERLAP_TOLERANCE,
) -> list[ClipCandidate]:
    """Drop overlapping clips, keeping the highest-scored of each clash.

    Greedy by score (ties: earlier start first); the kept clips come
    back in the input order (discovery order is start order, review
    order is the model's preference order). The discovery prompt asks
    for more candidates than the target, so dropping an overlap still
    leaves the requested count when spare candidates exist.
    """
    ranked = sorted(candidates, key=lambda c: (-c.score, c.start))
    kept: list[ClipCandidate] = []
    for cand in ranked:
        overlaps = any(
            min(cand.end, other.end) - max(cand.start, other.start) > tolerance
            for other in kept
        )
        if not overlaps:
            kept.append(cand)
    order = {id(c): i for i, c in enumerate(candidates)}
    return sorted(kept, key=lambda c: order[id(c)])


#: Clip-structure block appended to the Discovery system prompt.
DISCOVERY_STRUCTURE = (
    "Clip structure — every clip must be a self-contained moment of "
    "{{targetDuration}} seconds (between {{minDuration}} and "
    "{{maxDuration}} seconds) with this shape:\n"
    "- Clips must not overlap each other — every clip's time range must "
    "be separate from the others.\n"
    "- Hook (0–3 s): the opening line that stops the scroll.\n"
    "- Lock-in zone (3–7 s): the reason to keep watching.\n"
    "- Body Block 1: the first part of the payoff.\n"
    "- Rehook 1: a fresh hook that re-grabs attention.\n"
    "- Rehook 2: a second re-hook before the end.\n"
    "- Body Block 3: the final part of the payoff.\n"
    "- Closing (optional): a clean ending line.\n"
    "- CTA: a short call to action (follow / comment).\n"
    "Return ONLY valid JSON, no other text:\n"
    '{"clips": [{"start": 12.5, "end": 54.0, "title": "…", '
    '"hookText": "…", "concludingText": "…"}]}'
)

#: Language code -> name used in the caption prompt.
LANGUAGE_NAMES = {
    "auto": "the clip's language",
    "bn": "Bengali",
    "en": "English",
    "ar": "Arabic",
    "ur": "Urdu",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "tr": "Turkish",
    "id": "Indonesian",
    "pt": "Portuguese",
    "ru": "Russian",
}


@dataclass
class ClipCandidate:
    """One proposed viral clip (timestamps in seconds)."""

    start: float
    end: float
    title: str = ""
    hook_text: str = ""
    concluding_text: str = ""
    score: float = 0.0
    reason: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


def render_template(template: str, variables: dict) -> str:
    """Replace ``{{name}}`` placeholders; unknown placeholders stay as-is."""
    out = template
    for key, value in variables.items():
        out = out.replace("{{" + key + "}}", str(value))
    return out


def format_transcript_for_llm(transcript: Transcript) -> str:
    """Render the transcript as timestamped lines for the model.

    Uses utterances when present, otherwise groups words into ~20 s blocks
    so the model still sees where each line falls in the video.
    """
    lines: list[str] = []
    if transcript.utterances:
        for utt in transcript.utterances:
            if utt.text:
                lines.append(f"[{utt.start:.1f}s–{utt.end:.1f}s] {utt.text}")
    else:
        block: list[Word] = []
        block_start = 0.0
        for word in transcript.words:
            if not block:
                block_start = word.start
            block.append(word)
            if word.end - block_start >= 20.0:
                lines.append(
                    f"[{block_start:.1f}s–{word.end:.1f}s] "
                    + " ".join(w.text for w in block)
                )
                block = []
        if block:
            lines.append(
                f"[{block_start:.1f}s–{block[-1].end:.1f}s] "
                + " ".join(w.text for w in block)
            )
    return "\n".join(lines)


def build_discovery_prompt(
    transcript: Transcript,
    *,
    clip_length: float = 40.0,
    target_clips: int = 5,
    requested_clips: int | None = None,
    excluded_ranges: list[tuple[float, float]] | None = None,
    template: str | None = None,
) -> tuple[str, str]:
    """Build the (system prompt, user message) pair for clip discovery.

    ``clip_length`` is the preferred clip duration; the allowed window is
    85–115% of it (so the default 40 s asks for 34–46 s clips — the
    40–45 s shape from the spec). ``searchBudget`` is deliberately larger
    than the target so the review step has options to choose from.
    """
    requested = requested_clips if requested_clips is not None else target_clips
    min_duration = max(5, int(round(clip_length * 0.85)))
    max_duration = int(round(clip_length * 1.15))
    excluded_instruction = ""
    if excluded_ranges:
        ranges = ", ".join(f"{s:.1f}–{e:.1f}s" for s, e in excluded_ranges)
        excluded_instruction = (
            f"\nDo not propose clips inside these already-chosen ranges: {ranges}."
        )
    variables = {
        "transcript": format_transcript_for_llm(transcript),
        "clipLength": clip_length,
        "targetClips": target_clips,
        "requestedClips": requested,
        "searchBudget": max(10, target_clips * 2),
        "targetCount": target_clips,
        "minDuration": min_duration,
        "maxDuration": max_duration,
        "targetDuration": int(round(clip_length)),
        "excludedRangesInstruction": excluded_instruction,
    }
    system = render_template(template or DEFAULT_DISCOVERY_PROMPT, variables)
    system += "\n\n" + render_template(DISCOVERY_STRUCTURE, variables)
    user = (
        "Transcript (timestamps in seconds):\n"
        + format_transcript_for_llm(transcript)
    )
    return system, user


def _extract_json(text: str):
    """Parse the first JSON object/array in ``text`` (fences tolerated)."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = min(
        (i for i in (cleaned.find("{"), cleaned.find("[")) if i >= 0),
        default=-1,
    )
    if start < 0:
        raise ProviderError("bad_response", "The AI provider returned no JSON")
    end_char = "}" if cleaned[start] == "{" else "]"
    end = cleaned.rfind(end_char)
    if end < start:
        raise ProviderError("bad_response", "The AI provider returned no JSON")
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ProviderError(
            "bad_response", "The AI provider returned invalid JSON"
        ) from exc


def _parse_clip_items(data, duration: float | None) -> list[ClipCandidate]:
    """Validate raw model output into clip candidates.

    Accepts ``{"clips": [...]}``, ``{"moments": [...]}`` or a bare list.
    Ranges with end <= start are dropped; ranges are clamped to the video
    duration when it is known. Nothing is invented — a bad item is skipped.
    """
    if isinstance(data, dict):
        items = data.get("clips", data.get("moments", []))
    elif isinstance(data, list):
        items = data
    else:
        items = []
    candidates: list[ClipCandidate] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start", item.get("start_seconds")))
            end = float(item.get("end", item.get("end_seconds")))
        except (TypeError, ValueError):
            continue
        if duration is not None:
            start = max(0.0, min(start, duration))
            end = max(0.0, min(end, duration))
        if end <= start:
            continue
        candidates.append(
            ClipCandidate(
                start=start,
                end=end,
                title=str(item.get("title", "") or ""),
                hook_text=str(item.get("hookText", item.get("hook_text", "")) or ""),
                concluding_text=str(
                    item.get("concludingText", item.get("concluding_text", "")) or ""
                ),
                score=float(item.get("score", 0.0) or 0.0),
                reason=str(item.get("reason", "") or ""),
            )
        )
    candidates.sort(key=lambda c: c.start)
    return candidates


def discover_clips(
    client: DeepSeekClient,
    transcript: Transcript,
    *,
    clip_length: float = 40.0,
    target_clips: int = 5,
    requested_clips: int | None = None,
    excluded_ranges: list[tuple[float, float]] | None = None,
    duration: float | None = None,
    template: str | None = None,
) -> list[ClipCandidate]:
    """Run the Discovery prompt and return validated clip candidates."""
    system, user = build_discovery_prompt(
        transcript,
        clip_length=clip_length,
        target_clips=target_clips,
        requested_clips=requested_clips,
        excluded_ranges=excluded_ranges,
        template=template,
    )
    content = client.chat_text(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        thinking=True,
    )
    video_duration = duration if duration is not None else transcript.duration
    return dedupe_overlapping(_parse_clip_items(_extract_json(content), video_duration))


def build_review_prompt(
    candidates: list[ClipCandidate],
    transcript: Transcript,
    *,
    target_count: int = 5,
    template: str | None = None,
) -> tuple[str, str]:
    """Build the (system prompt, user message) pair for the review step."""
    variables = {
        "transcript": format_transcript_for_llm(transcript),
        "clipLength": "",
        "targetClips": target_count,
        "requestedClips": target_count,
        "searchBudget": len(candidates),
        "targetCount": target_count,
        "minDuration": "",
        "maxDuration": "",
        "targetDuration": "",
        "excludedRangesInstruction": "",
    }
    system = render_template(template or DEFAULT_REVIEW_PROMPT, variables)
    proposed = [
        {
            "start": c.start,
            "end": c.end,
            "title": c.title,
            "hookText": c.hook_text,
            "concludingText": c.concluding_text,
        }
        for c in candidates
    ]
    user = (
        "Proposed clips (timestamps in seconds):\n"
        + json.dumps({"clips": proposed}, ensure_ascii=False)
        + "\n\nTranscript (timestamps in seconds):\n"
        + format_transcript_for_llm(transcript)
        + "\n\nReturn ONLY valid JSON with the surviving clips, same shape, "
        "timestamps unchanged."
    )
    return system, user


def review_clips(
    client: DeepSeekClient,
    candidates: list[ClipCandidate],
    transcript: Transcript,
    *,
    target_count: int = 5,
    template: str | None = None,
) -> list[ClipCandidate]:
    """Run the Review prompt; keep the strongest up to ``target_count``.

    Returned timestamps are matched back to the proposed candidates so the
    review can never move a cut — only keep or drop it. If the model
    returns unusable output, fall back to the top candidates by score.
    """
    if not candidates:
        return []
    system, user = build_review_prompt(
        candidates, transcript, target_count=target_count, template=template
    )
    try:
        content = client.chat_text(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            thinking=True,
        )
        reviewed = _parse_clip_items(_extract_json(content), None)
    except ProviderError:
        reviewed = []
    if reviewed:
        kept: list[ClipCandidate] = []
        for item in reviewed:
            match = next(
                (
                    c for c in candidates
                    if abs(c.start - item.start) <= 0.5
                    and abs(c.end - item.end) <= 0.5
                ),
                None,
            )
            if match is not None and match not in kept:
                kept.append(match)
        if kept:
            return dedupe_overlapping(kept)[:target_count]
    # Review output unusable — keep the best proposed clips by score.
    deduped = dedupe_overlapping(candidates)
    ranked = sorted(deduped, key=lambda c: (c.score, c.start), reverse=True)
    return sorted(ranked[:target_count], key=lambda c: c.start)


def generate_caption(
    client: DeepSeekClient,
    clip: ClipCandidate,
    transcript: Transcript,
    *,
    language: str = "auto",
) -> dict:
    """Write a TikTok caption + EXACTLY 4 hashtags for one clip.

    Thinking is disabled (fast, cheap). Returns
    ``{"caption": str, "hashtags": [str, str, str, str]}`` — when the model
    returns fewer than 4 hashtags, only the ones it gave are returned
    (nothing is invented); more than 4 are truncated to the first 4.
    """
    lang_name = LANGUAGE_NAMES.get(language, language)
    words = [w for w in transcript.words if w.start >= clip.start and w.end <= clip.end]
    snippet = " ".join(w.text for w in words) or clip.title
    system = (
        "You write TikTok captions. Write in "
        f"{lang_name}. "
        "Given the clip transcript, write one short, engaging caption "
        "(one or two sentences) and EXACTLY 4 relevant hashtags. "
        'Return ONLY valid JSON: {"caption": "…", "hashtags": ["#a", "#b", "#c", "#d"]}.'
    )
    user = (
        f"Clip ({clip.start:.1f}s–{clip.end:.1f}s, titled {clip.title!r}):\n{snippet}"
    )
    content = client.chat_text(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        thinking=False,
    )
    caption = ""
    hashtags: list[str] = []
    try:
        data = _extract_json(content)
        if isinstance(data, dict):
            caption = str(data.get("caption", "") or "").strip()
            raw_tags = data.get("hashtags", []) or []
            hashtags = [_normalise_hashtag(t) for t in raw_tags if str(t).strip()]
    except ProviderError:
        caption = content.strip()
    if not hashtags:
        # Model answered in prose — harvest any #tags it wrote.
        hashtags = [_normalise_hashtag(t) for t in re.findall(r"#\S+", content)]
    hashtags = [tag for tag in dict.fromkeys(hashtags) if tag][:4]
    return {"caption": caption, "hashtags": hashtags}


def _normalise_hashtag(tag: str) -> str:
    """Ensure a hashtag starts with ``#`` and has no trailing punctuation."""
    tag = str(tag).strip().rstrip(".,;:!?\"'")
    if not tag:
        return ""
    return tag if tag.startswith("#") else f"#{tag}"
