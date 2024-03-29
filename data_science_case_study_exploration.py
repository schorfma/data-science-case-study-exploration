"""Data Science Case Study Exploration Streamlit Script.

Authors:
    Michel Kaufmann,
    Martin Schorfmann
Since:
    2020-11-13
Version:
    2021-01-20
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

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# Definition of Global Variables

# Version date
VERSION = "2021-01-20"

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

ALTAIR_FONT_SIZE = 16

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


def separator():
    streamlit.markdown("---")


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

    logo_column, description_column = streamlit.columns(
        [1, 2]  # Second column will be twice as wide
    )

    logo_column.image(
        translation(f"libraries.{library_name}_logo"),
        use_column_width=True
    )

    description_column.markdown(
        paragraphs(
            "#### {library} `{name}`".format(
                library=translation("libraries.library"),
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


def confusion_values(
        true_positive: int,
        true_negative: int,
        false_positive: int,
        false_negative: int,
        container: streamlit._DeltaGenerator = streamlit._main
):
    """Explains confusion values and shows relative share.

    Args:
        true_positive:
            Number of True Positives.
        true_negative:
            Number of True Negatives.
        false_positive:
            Number of False Positives.
        false_negative:
            Number of False Negatives.
        container:
            Streamlit container to display the information in.
    """
    # pylint: disable=invalid-name
    data_count = sum([true_positive, true_negative, false_positive, false_negative])
    container.markdown(
        markdown_list(
            translation("data_classifier.data_count", count=data_count),
            *[
                f"`{key}` / **{name}**: `{value}` ({description})"
                for key, name, description, value in [
                    (
                        "tp",
                        translation("data_classifier.tp_name"),
                        translation("data_classifier.tp_description"),
                        true_positive
                    ),
                    (
                        "tn",
                        translation("data_classifier.tn_name"),
                        translation("data_classifier.tn_description"),
                        true_negative
                    ),
                    (
                        "fp",
                        translation("data_classifier.fp_name"),
                        translation("data_classifier.fp_description"),
                        false_positive
                    ),
                    (
                        "fn",
                        translation("data_classifier.fn_name"),
                        translation("data_classifier.fn_description"),
                        false_negative
                    )
                ]
            ]
        )
    )

def confusion_metrics(
        true_positive: int,
        true_negative: int,
        false_positive: int,
        false_negative: int,
        container: streamlit._DeltaGenerator = streamlit._main
):
    """Explains confusion metrics and shows their formulas.

    Args:
        true_positive:
            Number of True Positives.
        true_negative:
            Number of True Negatives.
        false_positive:
            Number of False Positives.
        false_negative:
            Number of False Negatives.
        container:
            Streamlit container to display the information in.
    """

    data_count = sum(
        [true_positive, true_negative, false_positive, false_negative]
    )

    (
        accuracy_column,
        precision_column,
        recall_column
    ) = container.columns(3)

    # Accuracy
    accuracy_column.markdown(
        "##### " + translation("data_classifier.accuracy")
    )

    accuracy_column.markdown(
        translation("data_classifier.accuracy_description")
    )

    accuracy = (true_positive + true_negative) / data_count

    accuracy_column.latex(
        r"\frac{TP + TN}{TP + TN + FP + FN} = " +
        f"{accuracy:.2f}"
    )

    accuracy_column.markdown(
        translation(
            "data_classifier.accuracy_numbers",
            all=data_count,
            correct_percentage=f"{accuracy * 100:.0f}",
            incorrect_percentage=f"{(1 - accuracy) * 100:.0f}"
        )
    )

    # Precision
    precision_column.markdown(
        "##### " + translation("data_classifier.precision")
    )

    precision_column.markdown(
        translation("data_classifier.precision_description")
    )

    precision = true_positive / (true_positive + false_positive)

    precision_column.latex(
        r"\frac{TP}{TP + FP} = " +
        (
            f"{precision:.2f}"
            if true_positive else "0.00"
        )
    )

    precision_column.markdown(
        translation(
            "data_classifier.precision_numbers",
            positive_prediction=(true_positive + false_positive),
            true_percentage=f"{precision * 100:.0f}",
            false_percentage=f"{(1 - precision) * 100:.0f}"
        )
    )

    # Recall
    recall_column.markdown(
        "##### " + translation("data_classifier.recall")
    )

    recall_column.markdown(
        translation("data_classifier.recall_description")
    )

    recall = true_positive / (true_positive + false_negative)

    recall_column.latex(
        r"\frac{TP}{TP + FN} = " + (
            f"{recall:.2f}"
            if true_positive else "0.00"
        )
    )

    recall_column.markdown(
        translation(
            "data_classifier.recall_numbers",
            positive_actual=(true_positive + false_negative),
            positive_percentage=f"{recall * 100:.0f}",
            negative_percentage=f"{(1 - recall) * 100:.0f}"
        )
    )

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
            source_url=(
                f"[{translation('sources.this_gitlab_title')}]"
                f"({translation('sources.this_gitlab_url')})"
            )
        ),
        translation(
            "common.live_available",
            live_url=(
                f"[{translation('sources.this_live_title')}]"
                f"({translation('sources.this_live_url')})"
            )
        )
    )
)

# Title of Web Application
streamlit.title(
    ":microscope: " + translation("common.title")
)

separator()

streamlit.header(
    translation("common.outline")
)

OUTLINE = [
    ":wave: " + translation("introduction.header"),
    ":floppy_disk: " + translation("data_access.header"),
    ":bar_chart: " + translation("data_visualization.header"),
    ":card_file_box: " + translation("data_classifier.header"),
    "👆 " + translation("compas_threshold.header")
]

(
    INTRODUCTION_HEADER,
    DATA_ACCESS_HEADER,
    DATA_VISUALIZATION_HEADER,
    DATA_CLASSIFIER_HEADER,
    COMPAS_THRESHOLD_HEADER
) = OUTLINE

streamlit.markdown(
    markdown_list(*OUTLINE, numbered=True)
)

streamlit.sidebar.subheader(
    translation("introduction.outline_header")
)

streamlit.sidebar.markdown(
    markdown_list(*OUTLINE, numbered=True)
)

separator()

streamlit.header(INTRODUCTION_HEADER)

streamlit.markdown(
    translation("introduction.subtitle")
)

separator()

streamlit.subheader(
    translation("introduction.terminology")
)

streamlit.markdown(
    translation("introduction.introduction")
)

(
    TERM_STATISTICS,
    TERM_DATA_SCIENCE,
    TERM_MACHINE_LEARNING
) = streamlit.columns(3)

TERM_STATISTICS.info(
    translation("introduction.term_statistics")
)

TERM_DATA_SCIENCE.info(
    translation("introduction.term_data_science")
)

TERM_DATA_SCIENCE.image(
    "./images/data-science-process.png",
    use_column_width=True
)

TERM_DATA_SCIENCE.markdown(
    "Mojassamehleiden, "
    "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0), "
    "[File:CRISPDM-Extended-Majid.png]"
    "(https://commons.wikimedia.org/wiki/File:CRISPDM-Extended-Majid.png) "
    "via Wikimedia Commons"
)

TERM_MACHINE_LEARNING.info(
    translation("introduction.term_machine_learning")
)

TERM_MACHINE_LEARNING.image(
    "images/MachineLearning-Context.png",
    use_column_width=True
)

separator()

streamlit.subheader("Case Study: _COMPAS_")

(
    COMPAS_INTRODUCTION,
    COMPAS_TERMS,
    COMPAS_ANIMATION
) = streamlit.columns(3)

COMPAS_INTRODUCTION.markdown(
    translation("introduction.compas_introduction")
)

COMPAS_TERMS.info(
    translation("introduction.term_compas")
)

COMPAS_TERMS.info(
    translation("introduction.term_recidivism")
)

COMPAS_ANIMATION.image(
    "https://wp.technologyreview.com/wp-content/uploads/2019/10/mit-alg-yb-02-7.gif",
    use_column_width=True,
    caption=translation("introduction.compas_animation_source")
)

separator()

# Database Access
streamlit.header(
    DATA_ACCESS_HEADER
)

(
    DATA_SOURCE_FIRST,
    DATA_SOURCE_SECOND,
    DATA_SOURCE_INFO
) = streamlit.columns(3)

DATA_SOURCE_INFO.info(
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

DATA_SOURCE_FIRST.markdown(
    translation("data_access.data_source_first")
)

DATA_SOURCE_SECOND.markdown(
    translation("data_access.data_source_second")
)

DATA_VARIABLE_CODE = streamlit.expander(
    translation("data_access.data_variable_code"),
    expanded=False
)

DATA_VARIABLE_CODE.code("import pandas")

DATA_VARIABLE_CODE.code(
    "DEFENDANTS_DATA: pandas.DataFrame"
)

DEFENDANTS_DATA: pandas.DataFrame

DEFENDANTS_DATA = pandas.read_sql_table(
    table_name="people",
    con=COMPAS_DATABASE_CONNECTION,
    index_col="id"
)

# Mapping Charge Degrees to ordinal classes
CHARGE_DEGREES: Dict[Text, int] = {
    "(F1)": 1,
    "(F2)": 2,
    "(F3)": 3,
    "(F4)": 4,
    "(F5)": 5,
    "(F6)": 6,
    "(F7)": 7,
    "(M1)": 8,
    "(M2)": 9,
    "(M3)": 10
}

DEFENDANTS_DATA = DEFENDANTS_DATA[
    DEFENDANTS_DATA.is_recid != -1
][
    DEFENDANTS_DATA.c_charge_degree.isin(list(CHARGE_DEGREES.keys()))
]

# Adding ordinal charge degree column
DEFENDANTS_DATA = DEFENDANTS_DATA.assign(
    charge_degree=DEFENDANTS_DATA.c_charge_degree.transform(
        lambda charge_degree_code: CHARGE_DEGREES.get(charge_degree_code)
    )
)

separator()

streamlit.subheader(
    translation("data_access.subheader_how_much_data")
)

DATA_SIZE_CODE = streamlit.expander(
    translation("data_access.data_size_code"),
    expanded=False
)


DATA_SIZE_CODE.code(
    "length_table = len(DEFENDANTS_DATA)"
)

length_table = len(DEFENDANTS_DATA)

streamlit.markdown(
    markdown_list(
        translation(
            "data_access.data_frame_number_rows",
            value=length_table
        )
    )
)

separator()

streamlit.subheader(
    translation("data_access.subheader_data_impression")
)

streamlit.markdown(
    markdown_list(
        translation("data_access.data_frame_display_not_all"),
        translation("data_access.data_frame_display_head")
    )
)

DATA_HEAD_CODE = streamlit.expander(
    translation("data_access.data_head_code"),
    expanded=False
)

DATA_HEAD_CODE.code(
    "head_of_table = DEFENDANTS_DATA.head()"
)

head_of_table = DEFENDANTS_DATA.head()

head_of_table.replace(to_replace="", value=None, inplace=True)

streamlit.dataframe(
    head_of_table
)

separator()

NUMBER_DATA_COLUMNS_CODE = streamlit.expander(
    translation("data_access.number_data_columns_code"),
    expanded=False
)

NUMBER_DATA_COLUMNS_CODE.code(
    "number_columns = len(DEFENDANTS_DATA.columns)"
)

number_columns = len(DEFENDANTS_DATA.columns)

streamlit.markdown(
    markdown_list(
        translation(
            "data_access.data_frame_number_columns",
            value=number_columns
        )
    )
)

COLUMNS_LIST_COLUMN, COLUMNS_EXPLANATION_COLUMN = streamlit.columns(2)

COLUMNS_LIST_COLUMN.markdown(
    "#### " + translation("data_access.defendants_column_list")
)

COLUMNS_LIST_COLUMN.write(
    list(DEFENDANTS_DATA.columns)
)

RELEVANT_COLUMNS = [
    "sex",
    "race",
    "age",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "priors_count",
    "charge_degree",
    "is_recid",
    "is_violent_recid"
]

COLUMNS_EXPLANATION_COLUMN.markdown(
    paragraphs(
        "#### " + translation("data_access.defendants_column_explanations"),
        markdown_list(
            *[
                f"`{column_name}`\n    - " +
                translation(
                    f"data_access.defendants_column_{column_name}"
                ) for column_name in RELEVANT_COLUMNS
            ]
        )
    )
)

separator()

streamlit.subheader(
    translation("data_access.subheader_describe")
)

streamlit.markdown(
    markdown_list(
        translation("data_access.data_frame_describe")
    )
)

DEFENDANTS_DATA = DEFENDANTS_DATA[
    RELEVANT_COLUMNS + ["decile_score"]
]

DESCRIBE_COLUMNS_CODE = streamlit.expander(
    translation("data_access.describe_columns_code"),
    expanded=False
)

DESCRIBE_COLUMNS_CODE.code(
    "table_description = DEFENDANTS_DATA.describe()"
)

table_description = DEFENDANTS_DATA.describe()

streamlit.dataframe(
    table_description
)

separator()

streamlit.header(
    DATA_VISUALIZATION_HEADER
)

streamlit.markdown(
    translation("data_visualization.intro")
)

separator()

streamlit.subheader(translation("data_visualization.correlation_matrix_label"))

streamlit.markdown(
    translation("data_visualization.data_correlation_intro")
)

correlation_columns = list(DEFENDANTS_DATA.columns)
correlation_columns.remove("decile_score")

defendants_data_certain_columns = DEFENDANTS_DATA[
    correlation_columns
]

correlation_gender = streamlit.radio(
    translation("data_visualization.correlation_according_to_genders"),
    options=[
        "All",
        "Female",
        "Male"
    ],
    format_func=lambda option: translation(f"data_visualization.gender_option_{option.lower()}")
)

if correlation_gender != "All":
    defendants_data_certain_columns = defendants_data_certain_columns[
        defendants_data_certain_columns.sex == correlation_gender
    ]

defendants_data_correlations = defendants_data_certain_columns.corr()

correlation_data = defendants_data_correlations.stack().reset_index().rename(
    columns={
        0: "correlation",
        "level_0": "first",
        "level_1": "second"
    }
)

correlation_data["correlation_label"] = correlation_data["correlation"].map(
    "{:.2f}".format
)

base_correlation_plot = altair.Chart(correlation_data).encode(
    x="second:O",
    y="first:O"
)

# Text layer with correlation labels
# Colors are for easier readability
correlation_plot_text = base_correlation_plot.mark_text(
    tooltip=altair.TooltipContent("encoding")
).encode(
    text="correlation_label",
    color=altair.condition(
        altair.datum.correlation > 0.5,
        altair.value("white"),
        altair.value("black")
    )
)

correlation_plot = base_correlation_plot.mark_rect(
    tooltip=altair.TooltipContent("encoding")
).encode(
    color=altair.Color(
        "correlation:Q",
        scale=altair.Scale(
            domain=[-1, 0, 1],
            range=["DarkBlue", "White", "DarkRed"],
            type="linear"
        )
    )
)

CORRELATION_MATRIX = correlation_plot + correlation_plot_text

(
    CORRELATION_MATRIX_COLUMN,
    CORRELATION_OBSERVATION_COLUMN
) = streamlit.columns([2, 1])

CORRELATION_MATRIX_COLUMN.altair_chart(
    CORRELATION_MATRIX.properties(
        width=600,
        height=600
    ).configure_axis(
        labelFontSize=ALTAIR_FONT_SIZE,
        titleFontSize=ALTAIR_FONT_SIZE
    ).configure_title(
        fontSize=ALTAIR_FONT_SIZE
    ).configure_legend(
        labelFontSize=ALTAIR_FONT_SIZE,
        titleFontSize=ALTAIR_FONT_SIZE
    ).configure_header(
        labelFontSize=ALTAIR_FONT_SIZE,
        titleFontSize=ALTAIR_FONT_SIZE
    )
)

AGE_RECID_EXPANDER = CORRELATION_OBSERVATION_COLUMN.expander(
    "age ↔ is_recid",
    expanded=False
)

AGE_RECID_EXPANDER.markdown(
    translation("data_visualization.age_recid_observation")
)

AGE_JUV_EXPANDER = CORRELATION_OBSERVATION_COLUMN.expander(
    "age ↔ juv_...",
    expanded=False
)

AGE_JUV_EXPANDER.markdown(
    translation("data_visualization.age_juv_observation")
)

AGE_PRIORS_COUNT_EXPANDER = CORRELATION_OBSERVATION_COLUMN.expander(
    "age ↔ priors_count",
    expanded=False
)

AGE_PRIORS_COUNT_EXPANDER.markdown(
    translation("data_visualization.age_priors_count_observation")
)

PRIORS_COUNT_RECID_EXPANDER = CORRELATION_OBSERVATION_COLUMN.expander(
    "priors_count ↔ is_recid",
    expanded=False
)

PRIORS_COUNT_RECID_EXPANDER.markdown(
    translation("data_visualization.priors_count_recid_observation")
)

RECID_VIOLENT_RECID_EXPANDER = CORRELATION_OBSERVATION_COLUMN.expander(
    "is_recid ↔ is_violent_recid",
    expanded=False
)
RECID_VIOLENT_RECID_EXPANDER.markdown(
    translation("data_visualization.recid_violent_recid_observation")
)

separator()

streamlit.subheader(
    translation("data_visualization.boxplot_chart")
)

age_bins_step = streamlit.radio(
    translation("data_visualization.select_age_category_step_label"),
    options=[5, 10, 15],
    format_func=lambda option: translation("data_visualization.years", count=f"{option:02d}")
)

BOXPLOT_CODE_EXPANDER = streamlit.expander(
    label=translation("data_visualization.boxplot_code_label"),
    expanded=False
)

BOXPLOT_CODE_EXPANDER.code(
    "import altair"
)

BOXPLOT_CODE_EXPANDER.code(
    f"""
boxplot_chart = altair.Chart(
    DEFENDANTS_DATA[
        # {translation("data_visualization.boxplot_only_needed_columns")}
        [
            "age",
            "priors_count",
            "is_recid",
            "sex"
        ]
    ]
).mark_boxplot().encode(
    x=altair.X("age:Q", bin=altair.Bin(step={age_bins_step})),
    y="priors_count:Q",
    color="is_recid:N",
    column="is_recid:N",
    row="sex:N"
)
    """
)

boxplot_chart = altair.Chart(
    DEFENDANTS_DATA[
        [
            "age",
            "priors_count",
            "is_recid",
            "sex"
        ]
    ]
).mark_boxplot(outliers=True).encode(
    x=altair.X("age:Q", bin=altair.Bin(step=age_bins_step)),
    y=altair.Y("priors_count:Q"),
    color="is_recid:N",
    column="is_recid:N",
    row="sex:N"
)

BOXPLOT_CHART_COLUMN, BOXPLOT_OBSERVATION_COLUMN = streamlit.columns([2, 1])

BOXPLOT_CHART_COLUMN.altair_chart(
    boxplot_chart.properties(
        width=300,
        height=200
    ).configure_axis(
        labelFontSize=ALTAIR_FONT_SIZE,
        titleFontSize=ALTAIR_FONT_SIZE
    ).configure_title(
        fontSize=ALTAIR_FONT_SIZE
    ).configure_legend(
        labelFontSize=ALTAIR_FONT_SIZE,
        titleFontSize=ALTAIR_FONT_SIZE
    ).configure_header(
        labelFontSize=ALTAIR_FONT_SIZE,
        titleFontSize=ALTAIR_FONT_SIZE
    )
)

BOXPLOT_OBSERVATION_COLUMN.markdown(
    translation("data_visualization.boxplot_observations")
)

separator()

# Training Recidivism Classifier
streamlit.header(
    DATA_CLASSIFIER_HEADER
)

INPUT_DATA_FEATURES = [
    "age",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "priors_count",  # Prior Convictions
    "charge_degree"  # 1 (First Degree Felony Charge) - 10 (Misdemeanour)
]

CATEGORIES_COLUMNS = [
    "sex",
    "race"
]

FEATURE_SELECTION_COLUMN, LABEL_SELECTION_COLUMN = streamlit.columns(2)

FEATURE_SELECTION_COLUMN.image(
    "./images/Classifier-DefendantData-Left.png",
    use_column_width=True
)

LABEL_SELECTION_COLUMN.image(
    "./images/Classifier-DefendantData-Right.png",
    use_column_width=True
)

SELECTED_INPUT_DATA_FEATURES = FEATURE_SELECTION_COLUMN.multiselect(
    translation("data_classifier.select_features"),
    options=INPUT_DATA_FEATURES + CATEGORIES_COLUMNS,
    default=INPUT_DATA_FEATURES
)

SELECTED_INPUT_DATA_LABEL = LABEL_SELECTION_COLUMN.radio(
    translation("data_classifier.select_label"),
    options=[
        "is_recid",
        "is_violent_recid"
    ]
)

streamlit.markdown(
    markdown_list(
        translation("data_classifier.train_feature_selection", label=SELECTED_INPUT_DATA_LABEL)
    )
)

INPUT_DATA = DEFENDANTS_DATA[
    SELECTED_INPUT_DATA_FEATURES
]

SELECTED_CATEGORIES_COLUMNS = set(
    SELECTED_INPUT_DATA_FEATURES
) - set(INPUT_DATA_FEATURES)

INPUT_DATA = pandas.get_dummies(
    INPUT_DATA,
    columns=list(SELECTED_CATEGORIES_COLUMNS)
)

LABEL_DATA = DEFENDANTS_DATA[
    [
        SELECTED_INPUT_DATA_LABEL
    ]
]

separator()

streamlit.subheader(
    translation("data_classifier.train_test_split")
)

TRAIN_PERCENTAGE, TEST_PERCENTAGE = (75, 25)

streamlit.markdown(
    markdown_list(
        translation(
            "data_classifier.train_test_split_explanation",
            train_percentage=TRAIN_PERCENTAGE,
            test_percentage=TEST_PERCENTAGE
        )
    )
)

TRAIN_TEST_SPLIT_CODE_EXPANDER = streamlit.expander(
    translation("data_classifier.train_test_split_code"),
    expanded=False
)

# pylint: disable=line-too-long
TRAIN_TEST_SPLIT_CODE_EXPANDER.code(
    f"""
from sklearn.model_selection import train_test_split

(
    input_train_data,  # {TRAIN_PERCENTAGE}% {translation("data_classifier.for_training")}
    input_test_data,   # {TEST_PERCENTAGE}% {translation("data_classifier.for_testing")}
    label_train_data,  # {TRAIN_PERCENTAGE}% {translation("data_classifier.for_training")}
    label_test_data    # {TEST_PERCENTAGE}% {translation("data_classifier.for_testing")}
) = train_test_split(
    input_data,  # {translation("data_classifier.feature_column", count=len(INPUT_DATA.columns))}: {", ".join(INPUT_DATA.columns)}
    label_data,  # {translation("data_classifier.target_column", count=len(LABEL_DATA.columns))}: {", ".join(LABEL_DATA.columns)}
    random_state=0
)
    """
)
# pylint: enable=line-too-long

(
    INPUT_TRAIN_DATA,
    INPUT_TEST_DATA,
    LABEL_TRAIN_DATA,
    LABEL_TEST_DATA
) = train_test_split(
    INPUT_DATA,
    LABEL_DATA,
    random_state=0
)

separator()

streamlit.subheader(
    translation("data_classifier.configuration_and_training")
)

DECISION_TREE_CLASSIFIER_CODE_EXPANDER = streamlit.expander(
    translation("data_classifier.decision_tree_classifier_training_code"),
    expanded=False
)

(
    DECISION_TREE_CLASSIFIER_TERM,
    DECISION_TREE_CLASSIFIER_WITH_CONFIGURATION,
    DECISION_TREE_CLASSIFIER_STRUCTURE
) = streamlit.columns(3)

DECISION_TREE_CLASSIFIER_TERM.info(
    translation("data_classifier.term_decision_tree")
)

DECISION_TREE_CLASSIFIER_WITH_CONFIGURATION.markdown(
    "#### " + translation("data_classifier.decision_tree")
)

DECISION_TREE_CLASSIFIER_WITH_CONFIGURATION.markdown(
    markdown_list(
        "Easy to understand and use",
        "Transparent view into how it arrives at a decision"
    )
)

DECISION_TREE_CLASSIFIER_WITH_CONFIGURATION.markdown(
    "#### " + translation("data_classifier.classifier_configuration_values")
)

max_leaf_nodes = DECISION_TREE_CLASSIFIER_WITH_CONFIGURATION.slider(
    translation("data_classifier.max_leaf_nodes"),
    min_value=3,
    max_value=20,
    value=5
)

DECISION_TREE_CLASSIFIER_CODE_EXPANDER.code(
    f"""
from sklearn.tree import DecisionTreeClassifier, export_text

classifier = DecisionTreeClassifier(
    max_leaf_nodes={max_leaf_nodes},
    random_state=0
)

classifier.fit(
    input_train_data,
    label_train_data
)

decision_tree_structure = export_text(classifier)
    """
)

ESTIMATOR = DecisionTreeClassifier(
    max_leaf_nodes=max_leaf_nodes,
    random_state=0
)

with streamlit.spinner("Fitting Classifier"):
    ESTIMATOR.fit(
        INPUT_TRAIN_DATA,
        LABEL_TRAIN_DATA
    )

    TREE_STRUCTURE_TEXT = export_text(ESTIMATOR)

    for index, column in reversed(list(enumerate(INPUT_DATA.columns))):
        TREE_STRUCTURE_TEXT = TREE_STRUCTURE_TEXT.replace(
            f"feature_{index}",
            column
        )
    if SELECTED_INPUT_DATA_LABEL == "is_violent_recid":
        REPLACE_LABELS = [
            "is_not_violent_recid",
            "is_violent_recid"
        ]
    else:
        REPLACE_LABELS = [
            "is_not_recid",
            "is_recid"
        ]

    for index, target in enumerate(REPLACE_LABELS):
        TREE_STRUCTURE_TEXT = TREE_STRUCTURE_TEXT.replace(
            f"class: {index}",
            f"class: {target}"
        )

    TREE_STRUCTURE_TEXT = TREE_STRUCTURE_TEXT.replace("<=", "≤")

    TREE_STRUCTURE_TEXT = TREE_STRUCTURE_TEXT.replace("> ", ">")

    DECISION_TREE_CLASSIFIER_STRUCTURE.markdown(
        "#### " + translation("data_classifier.decision_tree_structure")
    )

    DECISION_TREE_CLASSIFIER_STRUCTURE.code(
        TREE_STRUCTURE_TEXT,
        language=None
    )

LABEL_PREDICTION_DATA = ESTIMATOR.predict(INPUT_TEST_DATA)

separator()

streamlit.subheader(
    translation("data_classifier.metrics_and_interpretation")
)

CONFUSION_MATRIX = confusion_matrix(LABEL_TEST_DATA, LABEL_PREDICTION_DATA)

TEST_DATA_COUNT = len(INPUT_TEST_DATA)

tn, fp, fn, tp = CONFUSION_MATRIX.ravel()

streamlit.markdown(
    "#### " + translation("data_classifier.confusion_matrix_values")
)

CONFUSION_CODE_EXPANDER = streamlit.expander(
    label=translation("data_classifier.prediction_metrics_code")
)

CONFUSION_CODE_EXPANDER.code(
    """
from sklearn.metrics import confusion_matrix

predicted_label_test_data = classifier.predict(
    input_test_data
)

tn, fp, fn, tp = confusion_matrix(
    label_test_data,
    predicted_label_test_data
).ravel()
    """
)

(
    CONFUSION_VALUES,
    CONFUSION_VALUES_IMAGE
) = streamlit.columns([2, 1])

confusion_values(tp, tn, fp, fn, CONFUSION_VALUES)

CONFUSION_VALUES_IMAGE.image(
    "./images/ConfusionMetrics.png",
    use_column_width=True
)

CONFUSION_VALUES_IMAGE.markdown(
    "Walber, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0), via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Precisionrecall.svg)"
)

separator()

CONFUSION_METRICS = streamlit.container()

confusion_metrics(
    tp, tn, fp, fn, CONFUSION_METRICS.expander(
        translation("data_classifier.related_metrics"),
        expanded=True
    )
)

separator()

streamlit.header(
    COMPAS_THRESHOLD_HEADER
)

THRESHOLD_CHOOSING_BASE_DATA = DEFENDANTS_DATA[
    [
        "sex",
        "race",
        "decile_score",
        "is_recid",
        "is_violent_recid"
    ]
][
    DEFENDANTS_DATA.decile_score != -1
]

RACES = list(
    THRESHOLD_CHOOSING_BASE_DATA.race.unique()
)

(
    SAMPLE_RACE_COLUMN,
    SAMPLE_SIZE_COLUMN
) = streamlit.columns([2, 1])

SAMPLE_SIZE = SAMPLE_SIZE_COLUMN.slider(
    translation("compas_threshold.sample_size_slider"),
    min_value=100,
    max_value=500,
    step=100,
    value=100
)

RACES_SAMPLED_DATA = {
    race: THRESHOLD_CHOOSING_BASE_DATA[
        THRESHOLD_CHOOSING_BASE_DATA.race == race
    ].sample(n=SAMPLE_SIZE, replace=True, random_state=0)
    for race in RACES
}

# for race, race_sampled_data in RACES_SAMPLED_DATA.items():
#     streamlit.write(race)
#     streamlit.dataframe(
#         race_sampled_data
#     )
#     streamlit.dataframe(
#         race_sampled_data.describe()
#     )

RACES_SAMPLED_DATA_COMBINED = pandas.concat(
    list(RACES_SAMPLED_DATA.values())
)

SELECTED_RACES = SAMPLE_RACE_COLUMN.multiselect(
    translation(
        "compas_threshold.select_race_samples",
        sample_size=SAMPLE_SIZE
    ),
    options=RACES,
    default=[
        "Caucasian",
        "African-American"
    ]
)

for non_selected_race in list(
        set(RACES) - set(SELECTED_RACES)
):
    RACES_SAMPLED_DATA_COMBINED = RACES_SAMPLED_DATA_COMBINED[
        RACES_SAMPLED_DATA_COMBINED.race != non_selected_race
    ]

THRESHOLD = streamlit.slider(
    translation("compas_threshold.threshold_slider"),
    min_value=0,
    max_value=10,
    value=7
)

RACES_SAMPLED_DATA_COMBINED["action"] = (
    RACES_SAMPLED_DATA_COMBINED.decile_score > THRESHOLD
)

RACES_SAMPLED_DATA_COMBINED["correct"] = (
    RACES_SAMPLED_DATA_COMBINED.is_recid == RACES_SAMPLED_DATA_COMBINED.action
)

SHOW_CORRECT = streamlit.checkbox(
    translation("compas_threshold.show_correct_checkbox")
)

DECILE_SCORE_CHART_BASE = altair.Chart(
    RACES_SAMPLED_DATA_COMBINED
).mark_bar(
    tooltip=altair.TooltipContent("encoding")
).encode(
    x="decile_score:O",
    y="count(is_recid):Q",
    opacity=altair.Opacity(
        "action:N",
        scale=altair.Scale(
            domain=["Released", "Jailed"],
            range=[1.00, 0.50]
        )
        # sort="descending"
    ),
    color=altair.Color(
        "is_recid:N" if not SHOW_CORRECT
        else "correct:N",
        scale=altair.Scale(
            domain=["Not Recid", "Recid"]
        ) if not SHOW_CORRECT else altair.Scale(
            domain=["Incorrect", "Correct"],
            range=["Tomato", "LimeGreen"]
        )
    )
).transform_calculate(
    is_recid="{0: 'Not Recid', 1: 'Recid'}[datum.is_recid]",
    action="{0: 'Released', 1: 'Jailed'}[datum.action]",
    correct="{0: 'Incorrect', 1: 'Correct'}[datum.correct]"
).transform_window(
    x="rank()",
    groupby=["decile_score"]
)

DECILE_SCORE_CHART_ALL = DECILE_SCORE_CHART_BASE

DECILE_SCORE_CHART_RACES = DECILE_SCORE_CHART_BASE.encode(
    column=altair.Column("race:N", sort="ascending")
)

ALL_CHART_CONTAINER = streamlit.container()

ALL_CONFUSION_CONTAINER = streamlit.container()

ALL_CHART_CONTAINER.altair_chart(
    DECILE_SCORE_CHART_ALL.configure_axis(
        labelFontSize=ALTAIR_FONT_SIZE,
        titleFontSize=ALTAIR_FONT_SIZE
    ).configure_title(
        fontSize=ALTAIR_FONT_SIZE
    ).configure_legend(
        labelFontSize=ALTAIR_FONT_SIZE,
        titleFontSize=ALTAIR_FONT_SIZE
    ).configure_header(
        labelFontSize=ALTAIR_FONT_SIZE,
        titleFontSize=ALTAIR_FONT_SIZE
    )
)

ALL_COUNT = len(
    RACES_SAMPLED_DATA_COMBINED
)

ALL_TN, ALL_FP, ALL_FN, ALL_TP = confusion_matrix(
    RACES_SAMPLED_DATA_COMBINED.is_recid,
    RACES_SAMPLED_DATA_COMBINED.action
).ravel()

confusion_values(
    ALL_TP, ALL_TN, ALL_FP, ALL_FN,
    ALL_CONFUSION_CONTAINER.expander(
        translation("data_classifier.confusion_matrix_values"),
        expanded=False
    )
)

confusion_metrics(
    ALL_TP, ALL_TN, ALL_FP, ALL_FN,
    ALL_CONFUSION_CONTAINER.expander(
        translation("data_classifier.related_metrics"),
        expanded=False
    )
)

separator()

streamlit.subheader(
    translation("compas_threshold.split_ethnicity")
)

RACES_CHART_CONTAINER = streamlit.container()
RACES_CONFUSION_CONTAINER = streamlit.container()

RACES_CHART_CONTAINER.altair_chart(
    DECILE_SCORE_CHART_RACES.configure_axis(
        labelFontSize=ALTAIR_FONT_SIZE,
        titleFontSize=ALTAIR_FONT_SIZE
    ).configure_title(
        fontSize=ALTAIR_FONT_SIZE
    ).configure_legend(
        labelFontSize=ALTAIR_FONT_SIZE,
        titleFontSize=ALTAIR_FONT_SIZE
    ).configure_header(
        labelFontSize=ALTAIR_FONT_SIZE,
        titleFontSize=ALTAIR_FONT_SIZE
    )
)

for race in sorted(SELECTED_RACES):
    race_tn, race_fp, race_fn, race_tp = confusion_matrix(
        RACES_SAMPLED_DATA_COMBINED[
            RACES_SAMPLED_DATA_COMBINED.race == race
        ].is_recid,
        RACES_SAMPLED_DATA_COMBINED[
            RACES_SAMPLED_DATA_COMBINED.race == race
        ].action
    ).ravel()

    RACES_CONFUSION_CONTAINER.markdown(f"#### `{race}`")

    confusion_values(
        race_tp, race_tn, race_fp, race_fn,
        RACES_CONFUSION_CONTAINER.expander(
            translation("data_classifier.confusion_matrix_values"),
            expanded=False
        )
    )

    confusion_metrics(
        race_tp, race_tn, race_fp, race_fn,
        RACES_CONFUSION_CONTAINER.expander(
            translation("data_classifier.related_metrics"),
            expanded=False
        )
    )

separator()

streamlit.subheader(
    translation("compas_explanation.ethical_view_header")
)

ETHICAL_VIEW_COLUMN, PREDICTION_FAILS_COLUMN = streamlit.columns(2)

ETHICAL_VIEW_COLUMN.markdown(
    translation("compas_explanation.ethical_view")
)

PREDICTION_FAILS_COLUMN.image(
    "images/COMPAS-Prediction-Fails-Differently.png",
    use_column_width=True
)

separator()

streamlit.header(
    ":books: " +
    translation("sources.header")
)

streamlit.info(
    translation("libraries.streamlit_intro")
)

STREAMLIT_LOGO_COLUMN, STREAMLIT_DESCRIPTION_COLUMN = show_library_two_columns(
    "streamlit"
)

PANDAS_LOGO_COLUMN, PANDAS_DESCRIPTION_COLUMN = show_library_two_columns(
    "pandas"
)

ALTAIR_LOGO_COLUMN, ALTAIR_DESCRIPTION_COLUMN = show_library_two_columns(
    "altair"
)

(
    SCIKIT_LEARN_LOGO_COLUMN,
    SCIKIT_LEARN_DESCRIPTION_COLUMN
) = show_library_two_columns(
    "scikit_learn"
)

SCIKIT_LEARN_DESCRIPTION_COLUMN.markdown(
    translation("libraries.scikit_learn_algorithm_cheat_sheet")
)

SOURCES = [
    "propublica_article_machine_bias",
    "propublica_article_compas_analysis",
    "propublica_github_compas_analysis",
    "technology_review_ai_fairer_judge",
    "northpointe_compas_faq"
]

SOURCE_ATTRIBUTES = [
    "url",
    "title",
    "subtitle",
    "organization",
    "authors",
    "date"
]

for source in SOURCES:
    (
        source_url,
        source_title,
        source_subtitle,
        source_organization,
        source_authors,
        source_date
    ) = (
        translation(f"sources.{source}_{attribute}")
        for attribute in SOURCE_ATTRIBUTES
    )



    streamlit.subheader(
        f"_{source_organization}_, "
        f"`{source_date}`: "
        f"{source_title}"
    )

    SOURCE_INFO_COLUMN, SOURCE_AUTHOR_COLUMN = streamlit.columns([2, 1])

    with SOURCE_INFO_COLUMN:
        streamlit.markdown(
            f"> {source_subtitle}"
        )

        streamlit.markdown(
            f"<{source_url}>"
        )

    with SOURCE_AUTHOR_COLUMN:
        streamlit.info(
            f"{translation('common.created_by', authors=source_authors)}"
        )
