"""GUI language state + inline-pair translation (stdlib only, import-light).

The first GUI interface (source-select dialog) owns the language choice:
``set_language`` runs BEFORE ``MainWindow`` is constructed, and every screen
reads ``tr`` at construction/usage time — so module-level constants must
store ``(zh, en)`` pairs and translate at the call site, never at import.
Changing language later requires the 回到开始 relaunch (same lifetime rule
as the backend selector: the dialog is the only place run-scoped choices
are made).

Scope: GUI chrome only (labels, buttons, titles, tabs, dialogs). Technical
detail strings that originate inside the camera service's events (stage
detail text, error payloads) pass through untranslated.
"""

from __future__ import annotations

LANG_ZH = "zh"
LANG_EN = "en"
LANGUAGES: tuple[str, ...] = (LANG_ZH, LANG_EN)
DEFAULT_LANGUAGE = LANG_ZH

_ALIASES = {
    "zh": LANG_ZH,
    "zh-cn": LANG_ZH,
    "zh_cn": LANG_ZH,
    "chinese": LANG_ZH,
    "en": LANG_EN,
    "en-us": LANG_EN,
    "english": LANG_EN,
}

_current = DEFAULT_LANGUAGE


def normalize_language(value: str | None) -> str:
    """Return a validated language id (None/empty -> the zh default)."""
    if value is None:
        return DEFAULT_LANGUAGE
    text = str(value).strip().lower()
    if not text:
        return DEFAULT_LANGUAGE
    if text in _ALIASES:
        return _ALIASES[text]
    raise ValueError(
        f"unknown GUI language {value!r}; expected one of {LANGUAGES}"
    )


def set_language(value: str | None) -> str:
    """Set the process-wide GUI language; returns the normalized id."""
    global _current
    _current = normalize_language(value)
    return _current


def language() -> str:
    """The current GUI language id."""
    return _current


def tr(zh: str, en: str) -> str:
    """Pick the current language's variant of an inline (zh, en) pair."""
    return en if _current == LANG_EN else zh
