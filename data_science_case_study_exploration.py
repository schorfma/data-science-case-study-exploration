from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Text
)
from typing_extensions import Protocol

import i18n
import pandas
import sqlalchemy
import streamlit


# Definition of Global Variables

# Version date
VERSION = "2020-11-14"

# Path to directory containing the translation files
TRANSLATION_PATH = Path("./translations")

# Available languages (Language name for language code)
TRANSLATION_LANGUAGES: Dict[Text, Text] = {
    "en": "English",
    "de": "Deutsch"
}

# Default language to fall back to
FALLBACK_LANGUAGE = "en"

# ProPublica COMPAS Analysis Database Path
PROPUBLICA_COMPAS_DATABASE_PATH = Path(
    "./data/propublica-compas-analysis/compas.db"
)

NEW_LINE = "\n"
NEW_PARAGRAPH = NEW_LINE * 2


# Definition of Functions

paragraphs = lambda *paragraph_items: NEW_PARAGRAPH.join(paragraph_items)

def markdown_list(
        *items: Text,
        numbered: bool = False
) -> Text:
    """Turns a list of items into a Markdown bullet or numbered list.

    Args:
        *items:
            Any number of text items to combine to a Markdown list.
        numbered:
            Optionally create a numbered / ordered list.

    Returns:
        The created Markdown list.
    """

    bullets: Iterable[Text]

    if numbered:
        bullet = "{number}."
    else:
        bullet = "*"

    bullets = [
        bullet.format(number=number)
        for number in range(1, len(items) + 1)
    ]

    bulleted_items = [
        f"{bullet} {item}"
        for bullet, item in zip(bullets, items)
    ]

    return NEW_LINE.join(bulleted_items)


class TranslationFunction(Protocol):
    def __call__(
        self,
        key: Text,
        **kwargs: Any
    ) -> Text:
        pass

def load_translation(
        translation_path: Path,
        language: Text,
        fallback_language: Text
) -> TranslationFunction:
    """Loads the chosen translation of the app.

    Args:
        translation_path:
            Path to directory containing translation files.
        language:
            Language code of chosen language.
        fallback_language:
            The default language code to fall back to.

    Returns:
        The translation function accepting a textual key and returning the
        localized string.
    """

    # Default configuration values:
    #     https://github.com/danhper/python-i18n/blob/master/i18n/config.py

    # Keep localization in memory
    i18n.set("enable_memoization", True)

    # Output error for missing translation keys
    i18n.set("error_on_missing_translation", True)

    # Set the translation path
    i18n.load_path.append(translation_path)

    # Set the selected language
    i18n.set("locale", language)

    # Set the fallback language
    i18n.set("fallback", fallback_language)

    return i18n.t


# Streamlit Script

# Language Selection Widget

LANGUAGE = streamlit.sidebar.selectbox(
    label="🌍 Select Language",
    options=list(TRANSLATION_LANGUAGES.keys()),
    format_func=TRANSLATION_LANGUAGES.get
)

# Translation Function to get localized Strings
TRANSLATION = load_translation(
    TRANSLATION_PATH,
    language=LANGUAGE,
    fallback_language=FALLBACK_LANGUAGE
)

streamlit.sidebar.info(
    paragraphs(
        TRANSLATION(
            "common.version",
            version=VERSION
        ),
        TRANSLATION(
            "common.created_by",
            authors=TRANSLATION("sources.this_authors")
        ),
        TRANSLATION(
            "common.source_available",
            source_url=TRANSLATION("sources.this_gitlab_url")
        )
    )
)

# Title of Web Application
streamlit.title(
    TRANSLATION("common.title")
)

# Hello World Message
streamlit.write(
    TRANSLATION("common.hello_world")
)

# Database Access
streamlit.header(
    "TODO: Accessing Database"
)

COMPAS_DATABASE_CONNECTION: sqlalchemy.engine.Connectable = sqlalchemy.create_engine(
    "sqlite:///" + PROPUBLICA_COMPAS_DATABASE_PATH.as_posix()
)

# Show first few rows for each COMPAS database table
for database_table_name in [
    "casearrest",
    "charge",
    "compas",
    "jailhistory",
    "people",
    "prisonhistory"
]:
    streamlit.subheader(
        f"Table `{database_table_name}`"
    )

    DATABASE_TABLE_DATA = pandas.read_sql_table(
        table_name=database_table_name,
        con=COMPAS_DATABASE_CONNECTION
    )

    streamlit.dataframe(
        DATABASE_TABLE_DATA.head()
    )
