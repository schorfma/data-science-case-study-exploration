"""Data Science Case Study Exploration Streamlit Script.

Authors:
    Michel Kaufmann,
    Martin Schorfmann
Since:
    2020-11-13
Version:
    2021-01-08
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
VERSION = "2021-01-08"

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

    # Accuracy
    container.markdown(
        "##### " + translation("data_classifier.accuracy")
    )

    container.markdown(
        translation("data_classifier.accuracy_description")
    )

    accuracy = (true_positive + true_negative) / data_count

    container.latex(
        "\\frac{TP + TN}{TP + TN + FP + FN} = " +
        f"{accuracy:.2f}"
    )

    container.markdown(
        translation(
            "data_classifier.accuracy_numbers",
            all=data_count,
            correct_percentage=f"{accuracy * 100:.0f}",
            incorrect_percentage=f"{(1 - accuracy) * 100:.0f}"
        )
    )

    # Precision
    container.markdown(
        "##### " + translation("data_classifier.precision")
    )

    container.markdown(
        translation("data_classifier.precision_description")
    )

    precision = true_positive / (true_positive + false_positive)

    container.latex(
        "\\frac{TP}{TP + FP} = " +
        (
            f"{precision:.2f}"
            if true_positive else "0.00"
        )
    )

    container.markdown(
        translation(
            "data_classifier.precision_numbers",
            positive_prediction=(true_positive + false_positive),
            true_percentage=f"{precision * 100:.0f}",
            false_percentage=f"{(1 - precision) * 100:.0f}"
        )
    )

    # Recall
    container.markdown(
        "##### " + translation("data_classifier.recall")
    )

    container.markdown(
        translation("data_classifier.recall_description")
    )

    recall = true_positive / (true_positive + false_negative)

    container.latex(
        "\\frac{TP}{TP + FN} = " + (
            f"{recall:.2f}"
            if true_positive else "0.00"
        )
    )

    container.markdown(
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

streamlit.header(
    ":wave: " + translation("introduction.header")
)

INTRODUCTION_COLUMN, PROCESS_COLUMN = streamlit.beta_columns([2, 1])

INTRODUCTION_COLUMN.markdown(
    translation("introduction.introduction")
)

INTRODUCTION_COLUMN.subheader(
    translation("introduction.outline_header")
)

OUTLINE = [
    ":floppy_disk: " + translation("data_access.header"),
    ":bar_chart: " + translation("data_visualization.header"),
    ":card_file_box: " + translation("data_classifier.header"),
    "🧭 " + translation("compas_explanation.header"),
    "👆 " + translation("compas_threshold.header")
]

(
    DATA_ACCESS_HEADER,
    DATA_VISUALIZATION_HEADER,
    DATA_CLASSIFIER_HEADER,
    COMPAS_EXPLANATION_HEADER,
    COMPAS_THRESHOLD_HEADER
) = OUTLINE

INTRODUCTION_COLUMN.markdown(
    markdown_list(*OUTLINE, numbered=True)
)

streamlit.sidebar.subheader(
    translation("introduction.outline_header")
)

streamlit.sidebar.markdown(
    markdown_list(*OUTLINE, numbered=True)
)

PROCESS_COLUMN.image(
    "./images/data-science-process.png",
    use_column_width=True
)

PROCESS_COLUMN.markdown(
    "Mojassamehleiden, "
    "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0), "
    "[File:CRISPDM-Extended-Majid.png]"
    "(https://commons.wikimedia.org/wiki/File:CRISPDM-Extended-Majid.png) "
    "via Wikimedia Commons"
)

# Database Access
streamlit.header(
    DATA_ACCESS_HEADER
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
    length_table = len(DEFENDANTS_DATA)

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
    head_of_table = DEFENDANTS_DATA.head()

streamlit.dataframe(
    head_of_table
)

with streamlit.echo():
    number_columns = len(DEFENDANTS_DATA.columns)

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

with streamlit.echo():
    table_description = DEFENDANTS_DATA.describe()

streamlit.dataframe(
    table_description
)

streamlit.header(
    DATA_VISUALIZATION_HEADER
)

streamlit.markdown(
    translation("data_visualization.intro")
)

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
) = streamlit.beta_columns([2, 1])

CORRELATION_MATRIX_COLUMN.altair_chart(
    CORRELATION_MATRIX.properties(
        width=600,
        height=600
    )
)

CORRELATION_OBSERVATION_COLUMN.markdown(
    translation("data_visualization.correlation_observations")
)

ALTAIR_LOGO_COLUMN, ALTAIR_DESCRIPTION_COLUMN = show_library_two_columns(
    "altair"
)

streamlit.code(
    "import altair"
)

age_bins_step = streamlit.radio(
    translation("data_visualization.select_age_category_step_label"),
    options=[5, 10, 15],
    format_func=lambda option: translation("data_visualization.years", count=f"{option:02d}")
)

BOXPLOT_CODE_EXPANDER = streamlit.beta_expander(
    label=translation("data_visualization.boxplot_code_label"),
    expanded=True
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
).mark_boxplot().encode(
    x=altair.X("age:Q", bin=altair.Bin(step=age_bins_step)),
    y="priors_count:Q",
    color="is_recid:N",
    column="is_recid:N",
    row="sex:N"
)

BOXPLOT_CHART_COLUMN, BOXPLOT_OBSERVATION_COLUMN = streamlit.beta_columns([2, 1])

BOXPLOT_CHART_COLUMN.altair_chart(
    boxplot_chart.properties(
        width=300,
        height=300
    )
)

BOXPLOT_OBSERVATION_COLUMN.markdown(
    translation("data_visualization.boxplot_observations")
)

# Training Recidivism Classifier
streamlit.header(
    DATA_CLASSIFIER_HEADER
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


FEATURE_SELECTION_CONTAINER = streamlit.beta_container()

FEATURE_SELECTION_CONTAINER.markdown(
    markdown_list(
        translation("data_classifier.train_implemented_classifier")
    )
)

FEATURE_SELECTION_COLUMN, LABEL_SELECTION_COLUMN = streamlit.beta_columns(2)

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

FEATURE_SELECTION_CONTAINER.markdown(
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

TRAIN_TEST_SPLIT_CODE_EXPANDER = streamlit.beta_expander(
    translation("data_classifier.train_test_split_code"),
    expanded=True
)

TRAIN_PERCENTAGE, TEST_PERCENTAGE = (75, 25)

TRAIN_TEST_SPLIT_CODE_EXPANDER.markdown(
    markdown_list(
        translation(
            "data_classifier.train_test_split_explanation",
            train_percentage=TRAIN_PERCENTAGE,
            test_percentage=TEST_PERCENTAGE
        )
    )
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

CLASSIFIER_CODE_COLUMN, MAX_LEAF_NODES_SLIDER_COLUMN = streamlit.beta_columns(2)

MAX_LEAF_NODES_SLIDER_COLUMN.markdown(
    "#### " + translation("data_classifier.classifier_configuration_values")
)

max_leaf_nodes = MAX_LEAF_NODES_SLIDER_COLUMN.slider(
    translation("data_classifier.max_leaf_nodes"),
    min_value=3,
    max_value=20,
    value=5
)

DECISION_TREE_CLASSIFIER_CODE_EXPANDER = CLASSIFIER_CODE_COLUMN.beta_expander(
    translation("data_classifier.decision_tree_classifier_training_code"),
    expanded=True
)

DECISION_TREE_CLASSIFIER_CODE_EXPANDER.markdown(
    markdown_list(
        translation("data_classifier.classifier_initialization"),
        translation("data_classifier.classifier_training")
    )
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

TREE_VIEW_COLUMN, CONFUSION_COLUMN, METRICS_COLUMN = streamlit.beta_columns(3)

ESTIMATOR = DecisionTreeClassifier(
    max_leaf_nodes=max_leaf_nodes,
    random_state=0
)

with streamlit.spinner():
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

    TREE_VIEW_COLUMN.markdown(
        "#### " + translation("data_classifier.decision_tree_structure")
    )

    TREE_VIEW_COLUMN.code(
        TREE_STRUCTURE_TEXT,
        language=None
    )

LABEL_PREDICTION_DATA = ESTIMATOR.predict(INPUT_TEST_DATA)

CONFUSION_MATRIX = confusion_matrix(LABEL_TEST_DATA, LABEL_PREDICTION_DATA)

TEST_DATA_COUNT = len(INPUT_TEST_DATA)

tn, fp, fn, tp = CONFUSION_MATRIX.ravel()

CONFUSION_COLUMN.markdown(
    "#### " + translation("data_classifier.confusion_matrix_values")
)

CONFUSION_CODE_EXPANDER = CONFUSION_COLUMN.beta_expander(
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

confusion_values(tp, tn, fp, fn, CONFUSION_COLUMN)

confusion_metrics(
    tp, tn, fp, fn, METRICS_COLUMN.beta_expander(
        translation("data_classifier.related_metrics"),
        expanded=True
    )
)

streamlit.header(
    COMPAS_EXPLANATION_HEADER
)

streamlit.write("TODO: Quotes?")

(
    COMPAS_EXPLANATION_COLUMN, COMPAS_ETHICAL_VIEW_COLUMN
) = streamlit.beta_columns(2)

with COMPAS_EXPLANATION_COLUMN:
    streamlit.subheader(
        translation("compas_explanation.explanation_compas_header")
    )

    streamlit.markdown(
        translation("compas_explanation.explanation_compas")
    )

with COMPAS_ETHICAL_VIEW_COLUMN:
    streamlit.subheader(
        translation("compas_explanation.ethical_view_header")
    )

    streamlit.markdown(
        translation("compas_explanation.ethical_view")
    )

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

SAMPLE_SIZE = streamlit.slider(
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

SELECTED_RACES = streamlit.multiselect(
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
    column=altair.Column(
        "action:N",
        sort="descending"
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
    row=altair.Row("race:N", sort="ascending")
)

ALL_CHART_COLUMN, ALL_CONFUSION_COLUMN = streamlit.beta_columns(2)

ALL_CHART_COLUMN.altair_chart(
    DECILE_SCORE_CHART_ALL
)

ALL_COUNT = len(
    RACES_SAMPLED_DATA_COMBINED
)

ALL_TN, ALL_FP, ALL_FN, ALL_TP = confusion_matrix(
    RACES_SAMPLED_DATA_COMBINED.is_recid,
    RACES_SAMPLED_DATA_COMBINED.action
).ravel()

confusion_values(ALL_TP, ALL_TN, ALL_FP, ALL_FN, ALL_CONFUSION_COLUMN)

confusion_metrics(ALL_TP, ALL_TN, ALL_FP, ALL_FN, ALL_CONFUSION_COLUMN)

RACES_CHART_COLUMN, RACES_CONFUSION_COLUMN = streamlit.beta_columns(2)

RACES_CHART_COLUMN.altair_chart(
    DECILE_SCORE_CHART_RACES
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

    RACES_CONFUSION_COLUMN.subheader(f"`{race}`")

    confusion_values(
        race_tp, race_tn, race_fp, race_fn,
        RACES_CONFUSION_COLUMN.beta_expander(
            translation("data_classifier.confusion_matrix_values"),
            expanded=False
        )
    )

    confusion_metrics(
        race_tp, race_tn, race_fp, race_fn,
        RACES_CONFUSION_COLUMN.beta_expander(
            translation("data_classifier.related_metrics"),
            expanded=True
        )
    )

streamlit.header(
    ":clipboard: " +
    translation("introduction.outline_recap_header")
)

OUTLINE_COMMENTS = [
    translation("data_access.summary"),
    translation("data_visualization.summary"),
    translation("data_classifier.summary"),
    translation("compas_explanation.summary"),
    translation("compas_threshold.summary")
]

OUTLINE_PREVIEW_ITEMS = [
    DEFENDANTS_DATA.head(n=3),
    CORRELATION_MATRIX.properties(width=400, height=300),
    f"![scikit-learn]({translation('libraries.scikit_learn_logo_url')})",
    f"#### {translation('compas_explanation.compas_acronym')} "
    f"({translation('compas_explanation.compas_acronym_full')})",
    DECILE_SCORE_CHART_ALL
]

for index, section, comment, preview in zip(
        range(1, len(OUTLINE) + 1),
        OUTLINE,
        OUTLINE_COMMENTS,
        OUTLINE_PREVIEW_ITEMS
):
    section_column, preview_column = streamlit.beta_columns(2)

    section_column.subheader(f"{index}. " + section)

    section_column.markdown(comment)

    preview_column.write(preview)

streamlit.header(
    ":books: " +
    translation("sources.header")
)

streamlit.info(
    translation("libraries.streamlit_intro")
)

STREAMLIT_LOGO_COLUMN, STREAMLIT_DESCRIPTION_COLUMN = show_library_two_columns("streamlit")

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

    SOURCE_INFO_COLUMN, SOURCE_AUTHOR_COLUMN = streamlit.beta_columns([2, 1])

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
