from pathlib import Path

import streamlit
import i18n

from typing import Callable, Dict, Text


TRANSLATIONS_PATH = Path("./translations")

TRANSLATION_LANGUAGES = {
    "en": "English",
    "de": "Deutsch"
}

FALLBACK_LANGUAGE = "en"

def load_translation(
        translations_path: Path,
        translation_languages: Dict[Text, Text],
        language: Text,
        fallback_language: Text
) -> Callable[[Text], Text]:
    i18n.set("enable_memoization", True)
    i18n.set("error_on_missing_translation", True)

    i18n.load_path.append(translations_path)

    i18n.set("locale", language)
    i18n.set("fallback", fallback_language)

    return i18n.t


LANGUAGE = streamlit.sidebar.selectbox(
    label="🌍 Select Language",
    options=list(TRANSLATION_LANGUAGES.keys()),
    format_func=TRANSLATION_LANGUAGES.get
)

TRANSLATION = load_translation(
    TRANSLATIONS_PATH,
    TRANSLATION_LANGUAGES,
    language=LANGUAGE,
    fallback_language=FALLBACK_LANGUAGE
)

streamlit.title(
    TRANSLATION("common.title")
)

streamlit.write(
    TRANSLATION("common.hello_world")
)
