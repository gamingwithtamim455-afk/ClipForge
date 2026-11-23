# Bundled caption fonts

All fonts are SIL Open Font License 1.1. Each family's license is kept
in `licenses/<family>-OFL.txt` (the shared `OFL.txt` in this directory
is the same license text).

Downloaded from the Google Fonts repository
(https://github.com/google/fonts, `ofl/` directory), except Montserrat,
which comes from its upstream repository
(https://github.com/JulietaUla/Montserrat, `fonts/ttf/`).

## Viral podcast fonts (listed first in the font picker)

| File | Family | Display |
|---|---|---|
| `Anton-Regular.ttf` | Anton | Anton |
| `BebasNeue-Regular.ttf` | Bebas Neue | Bebas Neue |
| `Montserrat-Black.ttf` | Montserrat Black | Montserrat Black |
| `Montserrat-Bold.ttf` | Montserrat | Montserrat Bold |
| `Oswald[wght].ttf` | Oswald | Oswald |
| `Poppins-Bold.ttf` | Poppins | Poppins Bold |
| `Poppins-ExtraBold.ttf` | Poppins ExtraBold | Poppins ExtraBold |
| `ArchivoBlack-Regular.ttf` | Archivo Black | Archivo Black |
| `BarlowCondensed-Bold.ttf` | Barlow Condensed | Barlow Condensed Bold |

## Language fonts

| File | Family | Used for |
|---|---|---|
| `NotoSansBengali.ttf` | Noto Sans Bengali | Bengali (`bn`) |
| `NotoNaskhArabic.ttf` | Noto Naskh Arabic | Arabic (`ar`) |
| `NotoNastaliqUrdu.ttf` | Noto Nastaliq Urdu | Urdu (`ur`) |
| `NotoSans.ttf` | Noto Sans | everything else (Latin etc.) |

Notes:

- Montserrat Bold's family name is "Montserrat" (the Bold weight is
  picked by fontconfig when the preset is bold); Montserrat Black is
  its own family, "Montserrat Black".
- `Oswald[wght].ttf` is a variable font; libass renders the default
  (Regular 400) instance.
- FFmpeg's `ass` filter is pointed at this directory with its `fontsdir`
  option (see `core/render.py`), so the fonts work even on machines that
  do not have them installed. When a custom font is installed, the
  user-fonts folder (app data dir) holds copies of these bundled fonts
  too, so it works as a single complete `fontsdir`.
