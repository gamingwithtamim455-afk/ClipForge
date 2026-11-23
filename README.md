# ClipForge

Turn long videos (podcasts, talks, sermons) into vertical viral clips —
transcribe, find the best moments, track the speaker's face, and render
9:16 clips with burned-in captions.

Free and open source. Python 3.12, PySide6 (Qt6). **Windows 11 is the
first target**; the same code runs on Linux for testing.

> **Status: working end-to-end (v0.2.2).** Home, wizard (Select Video →
> Viral Clips → Edit → Render), Settings with 7 tabs, transcription
> (Deepgram / local Whisper), AI clip discovery (DeepSeek), face
> tracking, single-pass rendering with burned captions, TikTok captions +
> 4 hashtags, manual timeline ranges, encrypted API keys with inline
> validation, updates + diagnostics, weekly backups and rotating logs.
> The UI mirrors the reference FaceClip recording (rebranded as
> ClipForge, no accounts, no Experimental tab).

## Quick start

### Windows

```powershell
git clone <repo-url> clipforge
cd clipforge
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python tools\download_ffmpeg.py --platform win
python main.py
```

### Linux

```bash
git clone <repo-url> clipforge
cd clipforge
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tools/download_ffmpeg.py --platform linux
python main.py
```

Headless smoke test (screenshots Home + wizard, exits 0):

```bash
CLIPFORGE_SMOKE_TEST=1 QT_QPA_PLATFORM=offscreen python main.py
```

Run the tests:

```bash
pip install pytest
pytest tests/ -q
```

## Build a Windows installer

The GitHub Actions workflow (`.github/workflows/build-windows.yml`) runs
on `windows-latest`: installs Python 3.12, pinned requirements, downloads
bundled FFmpeg, runs the tests + smoke test, builds `dist/ClipForge.exe`
with PyInstaller (`clipforge.spec`), then builds the Inno Setup installer
(`installer/clipforge.iss`) and uploads both as artifacts.

Locally on Windows:

```powershell
pip install -r requirements.txt
python tools\download_ffmpeg.py --platform win
pyinstaller clipforge.spec --noconfirm
iscc installer\clipforge.iss
```

## Dependencies (pinned)

| Package | Version | Purpose |
|---|---|---|
| PySide6 | 6.12.0 | Qt6 UI |
| faster-whisper | 1.2.1 | Local offline transcription (fallback) |
| opencv-python-headless | 5.1.0.95 | Video frame work |
| mediapipe | 1.2.3 | Face detection/landmarks |
| keyring | 25.7.0 | Encrypted API-key storage (DPAPI / Secret Service) |
| requests | 2.34.2 | Provider HTTP calls |
| numpy | 2.5.5 | Numeric work |
| pyinstaller | 6.22.3 | Windows .exe build |

Direct dependencies are pinned in `requirements.txt`. `requirements.lock`
(pip freeze output) pins every transitive dependency too, so a rebuild
1+ year from now installs the exact same versions:

```bash
pip install -r requirements.lock
```

Note: `mediapipe` pulls in `opencv-contrib-python` as a transitive
dependency alongside `opencv-python-headless`; both resolve to the same
`cv2` 5.1.0.95 on the verified install.

## Project structure

```
main.py                  # repo-root launcher shim -> clipforge.main
clipforge/
  main.py                # entry point (python -m clipforge.main)
  app.py                 # MainWindow: Home <-> wizard navigation, settings
  settings.py            # versioned QSettings store (no API keys here)
  i18n/                  # en/bn/ar/ur JSON translations (RTL for ar/ur)
  ui/                    # home, wizard, viral_clips, edit, render,
                         # settings_dialog, timeline, theme, widgets,
                         # splash (animated intro + sidebar branding)
  core/                  # ffmpeg, transcription, face_tracking, render,
                         # pipeline (end-to-end wiring), projects (v2 store),
                         # updates, backups, app_logging, thumbnails, paths,
                         # branding (swappable user photo)
  assets/branding/       # user-photo.png — Boss's portrait (swappable)
  providers/             # deepseek, deepgram, whisper_local + presets.json
tools/download_ffmpeg.py # static FFmpeg builds (Windows gyan.dev, Linux jvs)
installer/clipforge.iss  # Inno Setup installer script
.github/workflows/build-windows.yml
tests/                   # pytest smoke / UI (offscreen) / provider tests
docs/provider-research.md
```

## Branding, sharing & Muse link

- **Animated intro:** on launch, Boss's portrait
  (`clipforge/assets/branding/user-photo.png`) assembles shard-by-shard
  in blue crystal tones, like the portfolio website intro. Pure Qt
  animation — no video files, no extra dependencies. Click or any key
  skips it; it closes by itself after ~2 s. To swap the photo, replace
  that file (any image format works — it is loaded by content, not by
  extension) or set Settings key `branding_photo` to a file path; no
  code change needed. A missing photo falls back to a gradient + title
  and never blocks launch.
- **Sidebar branding:** the same portrait sits at the bottom of the
  Home sidebar with a gentle pulsing blue glow.
- **Share via WhatsApp (Render step):** there is no free official
  WhatsApp API and unofficial login libraries risk an account ban, so
  ClipForge never logs into WhatsApp. The button opens the finished
  clip's folder in the file manager (attach the video in WhatsApp
  yourself) and copies the caption + hashtags to the clipboard.
- **Open Muse:** a link in the Home sidebar footer opens
  https://muse.ai in the browser. There is no public API to embed
  Muse inside the app.

## Provider notes (verified 2026-11-22 — see docs/provider-research.md)

- **DeepSeek** (OpenAI-compatible): default model `deepseek-flash`
  (current), second option `deepseek-v4-pro`. `deepseek-v4-flash` is a
  **retired legacy alias** — never the default. Base URL has no `/v1`
  segment (`https://api.deepseek.com`). Model list loads from
  `GET {base}/models`; thinking enabled (`reasoning_effort: "high"`) for
  clip discovery, disabled for per-clip caption generation.
- **Deepgram**: model `nova-3` for **all 12 languages** (Auto, Bengali,
  English, Arabic, Urdu, Hindi, Spanish, French, Turkish, Indonesian,
  Portuguese, Russian). Auth header is `Authorization: Token <key>`.
  There is **no `word_timestamps` parameter** — word-level timestamps
  come back by default. Key validation via `GET /v1/projects`.
- **Local Whisper** (faster-whisper) is the offline/unsupported-language
  fallback; the status line says so explicitly when it is used.
- Canonical model IDs live in `clipforge/providers/presets.json` so a
  future provider rename can be fixed without a rebuild. API keys are
  stored in the system keyring, never in settings files or logs.

## License

MIT (planned). See LICENSE when added.
