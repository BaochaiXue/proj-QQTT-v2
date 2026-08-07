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
    def test_none_and_empty_default_to_zh(self) -> None:
        assert i18n.normalize_language(None) == i18n.LANG_ZH
        assert i18n.normalize_language("") == i18n.LANG_ZH

    def test_aliases(self) -> None:
        assert i18n.normalize_language("zh") == i18n.LANG_ZH
        assert i18n.normalize_language("ZH-CN") == i18n.LANG_ZH
        assert i18n.normalize_language("Chinese") == i18n.LANG_ZH
        assert i18n.normalize_language("en") == i18n.LANG_EN
        assert i18n.normalize_language("English") == i18n.LANG_EN

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown GUI language"):
            i18n.normalize_language("fr")


class TestTr:
    def test_tr_follows_language(self) -> None:
        i18n.set_language("zh")
        assert i18n.tr("你好", "hello") == "你好"
        i18n.set_language("en")
        assert i18n.tr("你好", "hello") == "hello"

    def test_set_language_returns_normalized(self) -> None:
        assert i18n.set_language("English") == i18n.LANG_EN
        assert i18n.language() == i18n.LANG_EN


class TestModuleConstantsArePairs:
    """Module-level GUI constants must store (zh, en) pairs, never a
    pre-translated string (the i18n contract: translate at usage time)."""

    def test_screens_constants(self) -> None:
        pytest.importorskip("PySide6")
        from demo_v7.gui import screens

        for _key, pair in screens.WARMUP_STAGE_PLAN:
            assert isinstance(pair, tuple) and len(pair) == 2
        for pair in screens._GENERATE_ROW_LABELS.values():
            assert isinstance(pair, tuple) and len(pair) == 2
        for _key, pair, _color in screens._SAMPLING_SOURCES:
            assert isinstance(pair, tuple) and len(pair) == 2
