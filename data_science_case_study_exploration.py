"""Data Science Case Study Exploration Streamlit Script.

Authors:
    Michel Kaufmann,
    Martin Schorfmann
Since:
    2020-11-13
Version:
    2020-11-16
"""

from pathlib import Path
from typing import (
    Any, Dict, Iterable, Text, Tuple
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
VERSION = "2020-11-16"

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


def show_library_two_columns(
        library_name: Text
) -> Tuple[Any, Any]:
    """Introduces a library using a two-column layout and i18n.

    Args:
        library_name:
            The name of the library being introduced.

    Returns:
        The two Streamlit columns (as a tuple) containing
        the library information.
    """

    logo_column, description_column = streamlit.beta_columns(
        [1, 2]  # Second column will be twice as wide
    )

    logo_column.image(
        translation(f"libraries.{library_name}_logo"),
        use_column_width=True
    )

    description_column.markdown(
        paragraphs(
            "#### `{name}`".format(
                name=translation(f"libraries.{library_name}_name")
            ),
            "<{url}>".format(
                url=translation(f"libraries.{library_name}_url")
            ),
            "> {description}".format(
                description=translation(f"libraries.{library_name}_description")
            )
        )
    )

    return logo_column, description_column


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

streamlit.header(
    ":wave: " + "TODO: Introduction and Outline"
)

# Database Access
streamlit.header(
    ":floppy_disk: " + translation("data_access.header")
)

streamlit.info(
    paragraphs(
        "### {}".format(translation("common.data_source", count=1)),
        "{platform} `{title}`: <{url}>".format(
            platform=translation("sources.propublica_github_compas_analysis_platform"),
            title=translation("sources.propublica_github_compas_analysis_title"),
            url=translation("sources.propublica_github_compas_analysis_url")
        ),
        "> {subtitle}".format(
            subtitle=translation("sources.propublica_github_compas_analysis_subtitle")
        )
    )
)

COMPAS_DATABASE_CONNECTION: sqlalchemy.engine.Connectable = sqlalchemy.create_engine(
    "sqlite:///" + PROPUBLICA_COMPAS_DATABASE_PATH.as_posix()
)

streamlit.markdown(
    markdown_list(
        translation("data_access.use_real_data_defendants_broward_county"),
        translation("data_access.simplicity_data_table")
    )
)

PANDAS_LOGO_COLUMN, PANDAS_DESCRIPTION_COLUMN = show_library_two_columns(
    "pandas"
)

streamlit.code("import pandas")

with streamlit.echo():
    CRIMINAL_PEOPLE_DATA: pandas.DataFrame

CRIMINAL_PEOPLE_DATA = pandas.read_sql_table(
    table_name="people",
    con=COMPAS_DATABASE_CONNECTION,
    index_col="id"
)

streamlit.markdown(
    markdown_list(
        translation("data_access.data_frame_inspection"),
        translation("data_access.data_frame_how_many_entries")
    )
)

streamlit.subheader(
    translation("data_access.subheader_how_much_data")
)

with streamlit.echo():
    length_table = len(CRIMINAL_PEOPLE_DATA)

streamlit.markdown(
    markdown_list(
        translation(
            "data_access.data_frame_number_rows",
            variable="length_table",
            value=length_table
        ),
        translation("data_access.data_frame_display_not_all"),
        translation("data_access.data_frame_display_head")
    )
)

streamlit.subheader(
    translation("data_access.subheader_data_impression")
)

with streamlit.echo():
    head_of_table = CRIMINAL_PEOPLE_DATA.head()

streamlit.dataframe(
    head_of_table
)

with streamlit.echo():
    number_columns = len(CRIMINAL_PEOPLE_DATA.columns)

streamlit.markdown(
    markdown_list(
        translation(
            "data_access.data_frame_number_columns",
            variable="number_columns",
            value=number_columns
        )
    )
)

COLUMNS_LIST_COLUMN, COLUMNS_EXPLANATION_COLUMN = streamlit.beta_columns(2)

COLUMNS_LIST_COLUMN.markdown(
    "#### " + translation("data_access.criminal_people_column_list")
)

COLUMNS_LIST_COLUMN.write(
    list(CRIMINAL_PEOPLE_DATA.columns)
)

COLUMNS_EXPLANATION_COLUMN.markdown(
    paragraphs(
        "#### " + translation("data_access.criminal_people_column_explanations"),
        markdown_list(
            *[
                f"`{column_name}`\n    - " +
                translation(
                    f"data_access.criminal_people_column_{column_name}"
                ) for column_name in [
                    "sex",
                    "race",
                    "age",
                    "juv_fel_count",
                    "juv_misd_count",
                    "juv_other_count",
                    "priors_count",
                    "c_charge_degree",
                    "is_recid",
                    "is_violent_recid"
                ]
            ]
        )
    )
)

streamlit.markdown(
    markdown_list(
        translation("data_access.data_frame_describe")
    )
)

with streamlit.echo():
    table_description = CRIMINAL_PEOPLE_DATA.describe()

streamlit.dataframe(
    table_description
)

streamlit.header(
    ":bar_chart: " + "TODO: Data Visualization"
)

ALTAIR_LOGO_COLUMN, ALTAIR_DESCRIPTION_COLUMN = show_library_two_columns(
    "altair"
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

streamlit.header(
    ":slot_machine: " +
    "TODO: Create and explain system to classify "
    "recidivism risk with scikit-learn"
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

streamlit.altair_chart(
    altair.Chart(
        CRIMINAL_PEOPLE_DATA[
            [
                "age",
                "juv_fel_count",
                "juv_misd_count",
                "juv_other_count",
                "priors_count",  # Prior Convictions
                "is_recid"
            ]
        ][
            CRIMINAL_PEOPLE_DATA.is_recid != -1
        ]
    ).mark_circle().encode(
        x=altair.X(altair.repeat("column"), type="quantitative"),
        y=altair.Y(altair.repeat("row"), type="quantitative"),
        color="is_recid:N"
    ).repeat(
        row=["age", "juv_fel_count", "priors_count"],
        column=["priors_count", "juv_fel_count", "age"]
    )
)



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

streamlit.subheader("TODO: What happens if we add `race` to the input data?")

streamlit.subheader("TODO: Interface for predicting risk for fictional people")

streamlit.header(
    "🧭 " + "TODO: Explanation of COMPAS"
)

streamlit.header(
    "👆 " +
    "TODO: COMPAS Interactive Threshold Choosing"
)

streamlit.header(
    ":books: " +
    "TODO: Streamlit und Quellen"
)
