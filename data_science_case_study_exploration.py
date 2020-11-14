"""Data Science Case Study Exploration Streamlit Script.

Authors:
    Michel Kaufmann,
    Martin Schorfmann
Since:
    2020-11-13
Version:
    2020-11-14
"""

from pathlib import Path
from typing import (
    Any, Dict, Iterable, Text
)
from typing_extensions import Protocol

import altair
import i18n
import pandas
import sqlalchemy
import streamlit

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


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
    """Used as type annotation for translation function of i18n."""
    # pylint: disable=too-few-public-methods

    def __call__(
            self,
            key: Text,
            **kwargs: Any
    ) -> Text:
        pass

@streamlit.cache()
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

streamlit.set_page_config(
    page_title="Data Science Case Study Exploration",
    # Page Icon: Image URL or Emoji string
    page_icon=":microscope:",
    # Layout: "centered" or "wide"
    layout="wide",
    # Initial Sidebar State: "auto", "expanded" or "collapsed"
    initial_sidebar_state="auto"
)

# Language Selection Widget
streamlit.sidebar.subheader(
    "🌍 Language"
)

LANGUAGE = streamlit.sidebar.selectbox(
    label="Select Language",
    options=list(TRANSLATION_LANGUAGES.keys()),
    format_func=TRANSLATION_LANGUAGES.get
)

# Translation Function to get localized Strings
translation = load_translation(
    TRANSLATION_PATH,
    language=LANGUAGE,
    fallback_language=FALLBACK_LANGUAGE
)

# Information Box
streamlit.sidebar.subheader(
    ":information_source: " + translation("common.info")
)

streamlit.sidebar.info(
    paragraphs(
        translation(
            "common.version",
            version=VERSION
        ),
        translation(
            "common.created_by",
            authors=translation("sources.this_authors")
        ),
        translation(
            "common.source_available",
            source_url=translation("sources.this_gitlab_url")
        )
    )
)

# Title of Web Application
streamlit.title(
    ":microscope: " + translation("common.title")
)

with streamlit.echo():
    # Hello World Message
    streamlit.write(
        translation("common.hello_world")
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
        # "casearrest",
        # "charge",
        # "compas",
        # "jailhistory",
        "people",
        # "prisonhistory"
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

    streamlit.dataframe(
        DATABASE_TABLE_DATA.describe()
    )

CRIMINAL_PEOPLE_DATA: pandas.DataFrame = pandas.read_sql_table(
    table_name="people",
    con=COMPAS_DATABASE_CONNECTION
)

streamlit.altair_chart(
    altair.Chart(
        CRIMINAL_PEOPLE_DATA[
            [
                "age",
                "decile_score"
            ]
        ]
    ).mark_boxplot().encode(
        x="age:Q",
        y="decile_score:O"
    )
)

streamlit.altair_chart(
    altair.Chart(
        CRIMINAL_PEOPLE_DATA[
            [
                "age",
                "decile_score",
                "priors_count"  # Prior Convictions
            ]
        ]
    ).mark_rect().encode(
        x="age:Q",
        y="priors_count:Q",
        color=altair.Color(field="decile_score", type="quantitative")
    )
)

streamlit.altair_chart(
    altair.Chart(
        CRIMINAL_PEOPLE_DATA[
            [
                "age_cat",
                "decile_score",
                "priors_count"  # Prior Convictions
            ]
        ]
    ).mark_bar().encode(
        x="decile_score:O",
        y="mean(priors_count):Q",
        column="age_cat:N"
    )
)

INPUT_DATA = CRIMINAL_PEOPLE_DATA[
    [
        "age",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",  # Prior Convictions
    ]
]

LABEL_DATA = CRIMINAL_PEOPLE_DATA[
        "is_recid"
]

INPUT_TRAIN_DATA, INPUT_TEST_DATA, LABEL_TRAIN_DATA, LABEL_TEST_DATA = train_test_split(
    INPUT_DATA,
    LABEL_DATA,
    random_state=0
)

ESTIMATOR = DecisionTreeClassifier(
    max_leaf_nodes=3,
    random_state=0
)

with streamlit.spinner():
    ESTIMATOR.fit(INPUT_TRAIN_DATA, LABEL_TRAIN_DATA)

LABEL_PREDICTION_DATA = ESTIMATOR.predict(INPUT_TEST_DATA)

CONFUSION_MATRIX = confusion_matrix(LABEL_TEST_DATA, LABEL_PREDICTION_DATA)

streamlit.write(
    CONFUSION_MATRIX
)

# TODO: Look into data and visualize it

# TODO: Create and explain system to classify recidivism risk with scikit-learn

# TODO: What happens if we add "race" to the input data?

# TODO: Interface for predicting risk for fictional people

# TODO: Explanation of COMPAS

# TODO: Interactive Threshold Choosing

