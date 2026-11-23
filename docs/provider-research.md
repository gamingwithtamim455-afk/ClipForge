# ClipForge — Provider Research (Deepgram + DeepSeek)

Checked: **2026-11-22** (all pages crawled 2026-11-22 ~19:50–19:51 UTC).
This file drives the provider adapters (`deepseek`, `deepgram`, `whisper`) and the
`provider-presets.json`. Do not hard-code model IDs anywhere else in the app —
the DeepSeek dropdown is filled from `GET {base}/models` at runtime.

---

## 1. Deepgram — speech-to-text

### 1.1 Endpoints and auth

| Item | Value |
|---|---|
| Transcription endpoint | `POST https://api.deepgram.com/v1/listen` |
| Key validation endpoint | `GET https://api.deepgram.com/v1/projects` |
| Auth header (both endpoints) | `Authorization: Token <key>` |
| Request body (listen) | Raw audio bytes, `Content-Type` set to the audio encoding (e.g. `audio/wav`); or JSON `{"url": "..."}` |
| Listen auth alternative | `Authorization: Bearer <JWT>` (grant-token flow only — **not** for plain API keys; always use `Token <key>`) |

### 1.2 Query parameters we will use

```
model=nova-3
language=<bcp47>            # one of the codes in §1.3; omit when auto-detecting
detect_language=true        # only for Auto-detect; omit language when set
smart_format=true
punctuate=true
utterances=true
```

Notes:

- **Word timestamps are NOT a query parameter.** There is no `word_timestamps`
  parameter in the Listen API. Word-level `start`/`end` are returned by default
  in `results.channels[0].alternatives[0].words`. `utterances=true` additionally
  returns utterance segments in `results.utterances`.
- `smart_format=true` already implies punctuation; sending `punctuate=true` as
  well is redundant but harmless.
- `keywords` is **not supported with Nova-3** — use `keyterm` instead if
  vocabulary boosting is ever needed.
- Auto-detect: `detect_language=true` (accepts `true`, or a list of candidate
  BCP-47 tags to bias detection). Alternative for Nova-3 multilingual:
  `language=multi` plus repeatable `language_hint=<code>`.

### 1.3 Language → model mapping

Source: Deepgram "Models & languages overview" (page updated 2026-11-16).
`nova-3` multilingual covers **all 12** languages in the app's dropdown, so
**no language in the list needs the faster-whisper fallback**. The fallback
still exists in the adapter for unknown/unsupported languages and for offline
use, but it is not triggered by any of these 12.

| App language | BCP-47 code | Deepgram support | Model | Fallback |
|---|---|---|---|---|
| Auto-detect | — (`detect_language=true`) | Yes | `nova-3` | faster-whisper only if detection fails |
| Bengali | `bn` | Yes (nova-3 multilingual) | `nova-3` | not needed |
| English | `en` | Yes (nova-3 multilingual; `en-US` also available) | `nova-3` | not needed |
| Arabic | `ar` | Yes (nova-3 multilingual) | `nova-3` | not needed |
| Urdu | `ur` | Yes (nova-3 multilingual) | `nova-3` | not needed |
| Hindi | `hi` | Yes (nova-3 multilingual) | `nova-3` | not needed |
| Spanish | `es` | Yes (nova-3 multilingual; `es-419` also available) | `nova-3` | not needed |
| French | `fr` | Yes (nova-3 multilingual; `fr-CA` also available) | `nova-3` | not needed |
| Turkish | `tr` | Yes (nova-3 multilingual) | `nova-3` | not needed |
| Indonesian | `id` | Yes (nova-3 multilingual) | `nova-3` | not needed |
| Portuguese | `pt` | Yes (nova-3 multilingual; `pt-BR`, `pt-PT` also available) | `nova-3` | not needed |
| Russian | `ru` | Yes (nova-3 multilingual) | `nova-3` | not needed |

`nova-3-general` is the general-purpose variant in the same model family; the
overview page lists it alongside `nova-3` for multilingual use. Use plain
`nova-3` (Deepgram's recommended model for recordings / multilingual audio).
If Deepgram ever removes a language from Nova-3, the adapter must detect the
resulting 400/empty transcript and fall back to local faster-whisper, and say
so in the status line.

### 1.4 Key validation — `GET https://api.deepgram.com/v1/projects`

- `200` → key valid ("Connected"). Costs nothing, spends no transcription credit.
- `401` (`INVALID_AUTH`, `"Invalid credentials"`) → invalid or expired key.
- `403` → key lacks the required scope/permissions. **Caveat:** Deepgram keys
  have scopes (`member`, `listen`, `speak`, `manage`, `usage`). A key scoped
  to `listen` only may be denied on the management endpoint (`/v1/projects`).
  If validation returns 403, **do not report the key as invalid** — fall back
  to a minimal test transcription request (a 1-second audio clip) and judge by
  that result instead.
- `402` (`Payment Required`, "insufficient credits") → documented by Deepgram
  as a **speech-to-text** error (transcription attempted without sufficient
  funds). It is **not** documented for `GET /v1/projects`; if it appears on a
  listen call it means "no balance — top up". Do not expect it from the
  projects endpoint.
- `429` → rate limit (too many requests); retry with exponential backoff.
- Error body shape: `{"err_code": "INVALID_AUTH", "err_msg": "Invalid credentials"}`.
- Key format hint (secondary source, not first-party docs): Deepgram API keys
  are 40-character lowercase hex with no fixed prefix.

### 1.5 Chunking (app-side rule, from the build spec)

Send audio only (mono 16 kHz) in chunks of at most 8 minutes with 5 s overlap;
merge word timestamps across chunk boundaries by offsetting each chunk's
timestamps by its start time and de-duplicating words inside the overlap window.
(Chunking is not a Deepgram requirement — it is our reliability rule.)

### Sources — Deepgram

- https://developers.deepgram.com/docs/models-languages-overview (updated 2026-11-16)
- https://developers.deepgram.com/reference/listen-file (updated 2026-11-13)
- https://developers.deepgram.com/docs/errors
- https://github.com/deepgram/skills/blob/HEAD/skills/api/references/listen.md (Deepgram's own reference summary)

---

## 2. DeepSeek — chat completions (OpenAI-compatible)

### 2.1 ⚠️ Model IDs — the brief is wrong, the docs say the opposite

The build brief states: "Valid DeepSeek model IDs are `deepseek-v4-flash` and
`deepseek-v4-pro` (default `deepseek-v4-flash`). `deepseek-flash` is INVALID."

**The live DeepSeek docs (checked 2026-11-22) say the exact opposite:**

| Model ID | Status per api-docs.deepseek.com |
|---|---|
| `deepseek-flash` | **Current name.** Served by DeepSeek-V4.1-Flash-1017. Use this. |
| `deepseek-v4-pro` | **Current name.** Served by DeepSeek-V4-Pro-0813. |
| `deepseek-v4-flash` | **Legacy alias, model retired.** Still accepted, but requests are served by DeepSeek-V4.1-Flash-1017 and billed at the Flash price. Do not use as the default. |
| `deepseek-v4-flash-vision-exp` | Legacy alias, retired (same treatment as above). |
| `deepseek-chat` | **Retired** (retired 2026-07-24 15:59 UTC per migration guide; no longer listed in DeepSeek docs). |
| `deepseek-reasoner` | **Retired** (same date; no longer listed). |

**Adapter rule:** default model = `deepseek-flash`; second option =
`deepseek-v4-pro`. Fill the dropdown from `GET {base}/models` at runtime so
any future rename is picked up automatically; keep an editable-text fallback.
Do **not** present `deepseek-v4-flash` as a current model — at most accept it
as a legacy alias that the server maps to `deepseek-flash`.
Both current models: 1M-token context, max output 384K tokens, JSON output,
tool calls, Responses API, Anthropic API. Vision: Flash yes, Pro no.
FIM completion: non-thinking mode only, both models.

### 2.2 Base URL and normalization

- Canonical base URL (OpenAI format): `https://api.deepseek.com`
  (Anthropic format, not used by ClipForge: `https://api.deepseek.com/anthropic`).
- DeepSeek paths have **no `/v1` segment** (unlike OpenAI).
- Normalization rule: trim whitespace → strip trailing `/` → strip a trailing
  `/v1` if present → the result is the base. Then append `/chat/completions`
  or `/models`.
  - `https://api.deepseek.com` → `https://api.deepseek.com`
  - `https://api.deepseek.com/` → `https://api.deepseek.com`
  - `https://api.deepseek.com/v1` → `https://api.deepseek.com`
  - `https://api.deepseek.com/v1/` → `https://api.deepseek.com`

### 2.3 `GET {base}/models` — list models (key validation + dropdown source)

- Method/path: `GET https://api.deepseek.com/models`
- Headers: `Authorization: Bearer <key>`
- Response (OpenAI-compatible):
  ```json
  {
    "object": "list",
    "data": [
      {
        "id": "deepseek-flash",
        "object": "model",
        "owned_by": "deepseek",
        "name": "DeepSeek Flash",
        "context_window": 1000000,
        "max_output_tokens": 384000,
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
        "effort": { "supported_levels": ["low", "high", "max"], "default_level": "high" }
      }
    ]
  }
  ```
- `effort.supported_levels` lists the values the model's `reasoning_effort`
  accepts; `none` (thinking off) is not included in that list. `data[].id`
  is what fills the Model ID dropdown.
- Error mapping for validation: `401` = invalid key; `402` = no balance;
  `404` = wrong base URL (or an unknown path); `429` = rate limit.

### 2.4 `POST {base}/chat/completions` — request shape

- Method/path: `POST https://api.deepseek.com/chat/completions`
- Headers: `Content-Type: application/json`, `Authorization: Bearer <key>`
- Body (from DeepSeek's own example):
  ```json
  {
    "model": "deepseek-flash",
    "messages": [
      { "role": "system", "content": "You are a helpful assistant." },
      { "role": "user", "content": "Hello!" }
    ],
    "thinking": { "type": "enabled" },
    "reasoning_effort": "high",
    "stream": false
  }
  ```
- `model` (required): `deepseek-flash` or `deepseek-v4-pro`.
- `stream`: set `true` for a streamed response, `false` otherwise.

### 2.5 Thinking / reasoning mode — which models, how to enable

- **Both** current models (`deepseek-flash`, `deepseek-v4-pro`) support
  thinking mode **and** non-thinking mode. Thinking is **enabled by default**.
- Enable/disable (OpenAI format), `thinking` object — takes precedence over
  `reasoning_effort` when present:
  - `{"thinking": {"type": "enabled"}}` → thinking mode on
  - `{"thinking": {"type": "disabled"}}` → thinking mode off
- Effort control, `reasoning_effort` (string):
  - `none` = thinking off; `low`, `high`, `max` = thinking on.
  - Default effort is `high`. Accepted aliases: `minimal` → `low`;
    `medium` / `xhigh` → `high`; `ultra` → `max`.
  - If `thinking` is omitted or `null`, the mode follows `reasoning_effort`;
    if `reasoning_effort` is also omitted, thinking is enabled at `high`.
- With the OpenAI SDK, pass the `thinking` parameter inside `extra_body`
  (it is not a native SDK field).
- In thinking mode the chain-of-thought is returned in `reasoning_content`
  alongside `content`. If a multi-turn conversation carries `tools`, prior
  turns' `reasoning_content` should be passed back (docs state a 400 is
  possible otherwise; the server has been observed to be lenient — pass it back
  anyway when tools are used).
- **Thinking mode does not support `temperature`, `presence_penalty`, or
  `frequency_penalty`.** Setting them raises no error but has no effect —
  omit them whenever thinking is on.
- `top_p`: `deepseek-flash` — only takes effect in thinking mode (effective
  range 0.95–1.0); fixed at 1.0 in non-thinking mode. `deepseek-v4-pro` —
  ignored in thinking mode; takes effect in non-thinking mode.
- **ClipForge usage rule:** clip discovery / viral-clip selection prompts →
  thinking enabled, `reasoning_effort: "high"`. Per-clip TikTok caption +
  hashtag generation → thinking disabled
  (`{"thinking": {"type": "disabled"}}`) for speed and cost.

### 2.6 Error mapping (DeepSeek)

| HTTP | Meaning for the status line |
|---|---|
| 401 | Invalid API key |
| 402 | No balance (top up at platform.deepseek.com) |
| 404 | Wrong base URL (or unknown model/path) |
| 429 | Rate limit — retry with exponential backoff |
| 5xx / timeout | Server/network error — retry with exponential backoff |
| No internet | Friendly offline error, app must not crash |

### Sources — DeepSeek

- https://api-docs.deepseek.com (Your First API Call)
- https://api-docs.deepseek.com/quick_start/pricing (Models & Pricing — model versions, thinking support, prices)
- https://api-docs.deepseek.com/api/list-models/ (List Models response shape)
- https://api-docs.deepseek.com/api/create-chat-completion/ (Chat Completions — `model` values, `thinking`, `reasoning_effort`)
- https://api-docs.deepseek.com/guides/thinking_mode/ (Thinking Mode toggle/effort)
- https://dev.to/jangwook_kim_e31e7291ad98/deepseek-v4-pro-and-v4-flash-migration-guide-and-api-setup-3bpb (deepseek-chat / deepseek-reasoner retirement date 2026-07-24 15:59 UTC — secondary source)
- https://www.promptfoo.dev/docs/providers/deepseek/ (confirms `deepseek-chat` / `deepseek-reasoner` retired; `deepseek-v4-flash` routes to `deepseek-flash` — secondary source)

---

## 3. Adapter design notes (for the builder)

1. **DeepSeek adapter:** base URL normalized per §2.2; key validated by
   `GET {base}/models`; model dropdown from `data[].id`; default
   `deepseek-flash`, fallback `deepseek-v4-pro`; thinking per §2.5.
   Never log keys; store encrypted (Windows DPAPI/keyring).
2. **Deepgram adapter:** `POST https://api.deepgram.com/v1/listen` with
   `Authorization: Token <key>`; model `nova-3`; per-language `language`
   code from §1.3 or `detect_language=true` for Auto-detect; params per §1.2.
   Key validated by `GET https://api.deepgram.com/v1/projects`, with the
   403-scope fallback to a 1-second test transcription (§1.4).
3. **Whisper adapter (local faster-whisper):** used when the user selects
   Local Whisper, when Deepgram reports a language unsupported, or when
   offline. Status line must say so explicitly.
4. **Provider presets JSON:** remote-updatable file holding the canonical
   model IDs (`deepseek-flash`, `deepseek-v4-pro`, `nova-3`), endpoint paths,
   and the language→model table, so a future rename is fixable without a
   rebuild. The values in this file are the initial contents of that JSON.
