"""Unit tests for the GUI language module (pure python; no Qt needed)."""

from __future__ import annotations

import pytest

from demo_v7.gui import i18n


@pytest.fixture(autouse=True)
def _restore_language():
    before = i18n.language()
    yield
    i18n.set_language(before)


class TestNormalizeLanguage:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            # None/empty fold to the zh default (app.py may pass either).
            (None, i18n.LANG_ZH),
            ("", i18n.LANG_ZH),
            # Aliases, incl. the mixed-case/suffixed spellings a human
            # writes in yaml.
            ("zh", i18n.LANG_ZH),
            ("ZH-CN", i18n.LANG_ZH),
            ("Chinese", i18n.LANG_ZH),
            ("en", i18n.LANG_EN),
            ("English", i18n.LANG_EN),
        ],
    )
    def test_aliases(self, value: str | None, expected: str) -> None:
        """Guards demo_v7/gui/i18n.py:40 -- ``str(value).strip().lower()``.

        Without the case/whitespace fold the alias table misses "ZH-CN" /
        "Chinese" / "English" and the failure is SILENT: config_default_language
        (demo_v7/app.py:283-294) swallows the resulting ValueError at app.py:292,
        so a ``session.language: "English"`` yaml quietly launches the GUI in zh.
        """
        assert i18n.normalize_language(value) == expected


class TestTr:
    def test_tr_follows_language(self) -> None:
        """Guards demo_v7/gui/i18n.py:64 -- ``return en if _current == LANG_EN
        else zh``, the single line every GUI string passes through (~200
        ``tr(zh, en)`` call sites across screens.py/main_window.py/app.py).
        A ``return zh`` (a language frozen at import) is exactly the regression
        the cbf43dd commit message warns about.

        set_language/language/tr are one state triple, so this also pins
        set_language's contract: it normalizes and returns the normalized id
        (i18n.py:50-54), and language() reports it rather than a constant.
        """
        assert i18n.set_language("zh") == i18n.LANG_ZH
        assert (i18n.language(), i18n.tr("你好", "hello")) == (i18n.LANG_ZH, "你好")
        assert i18n.set_language("English") == i18n.LANG_EN
        assert (i18n.language(), i18n.tr("你好", "hello")) == (i18n.LANG_EN, "hello")


class TestModuleConstantsArePairs:
    """Module-level GUI constants must store (zh, en) pairs, never a
    pre-translated string (the i18n contract: translate at usage time)."""

    def test_screens_constants(self) -> None:
        """Guards the two constants that are unpacked with ``tr(*pair)``:
        ``_GENERATE_ROW_LABELS`` at demo_v7/gui/screens.py:348
        (``self._timeline.setRowLabel("sp:generate", tr(*pair))``) and
        ``_GS_MESH_SURFACE_SUBROWS`` at demo_v7/gui/screens.py:365
        (``[(key, tr(*pair)) for key, pair in _GS_MESH_SURFACE_SUBROWS]``).
        A bare string there is a TypeError/garbled row at run time, not a
        mistranslation.
        """
        pytest.importorskip("PySide6")
        from demo_v7.gui import screens

        for pair in screens._GENERATE_ROW_LABELS.values():
            assert isinstance(pair, tuple) and len(pair) == 2
        for _key, pair in screens._GS_MESH_SURFACE_SUBROWS:
            assert isinstance(pair, tuple) and len(pair) == 2
