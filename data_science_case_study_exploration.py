from pathlib import Path

import i18n
import pandas
import sqlalchemy
import streamlit

from typing import Callable, Dict, Text


# Definition of Global Variables

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

# Definition of Functions

def load_translation(
        translation_path: Path,
        language: Text,
        fallback_language: Text
) -> Callable[[Text], Text]:
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
