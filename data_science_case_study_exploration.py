"""Data Science Case Study Exploration Streamlit Script.

Authors:
    Michel Kaufmann,
    Martin Schorfmann
Since:
    2020-11-13
Version:
    2020-11-19
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
VERSION = "2020-11-19"

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

INTRODUCTION_COLUMN, PROCESS_COLUMN = streamlit.beta_columns([2, 1])

INTRODUCTION_COLUMN.markdown(
    "TODO: Introduction Text"
)

INTRODUCTION_COLUMN.subheader("TODO: Outline")

OUTLINE = [
    ":floppy_disk: " + translation("data_access.header"),
    ":bar_chart: " + "TODO: Data Visualization",
    ":card_file_box: " + "TODO: Training a Recidivism Classifier",
    "🧭 " + "TODO: Explanation of COMPAS",
    "👆 " + "TODO: COMPAS Interactive Threshold Choosing"
]

INTRODUCTION_COLUMN.markdown(
    markdown_list(*OUTLINE, numbered=True)
)

streamlit.sidebar.subheader("TODO: Outline")

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
    "[File:CRISPDM-Extended-Majid.png](https://commons.wikimedia.org/wiki/File:CRISPDM-Extended-Majid.png) via Wikimedia Commons"
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

CRIMINAL_PEOPLE_DATA = CRIMINAL_PEOPLE_DATA[
    CRIMINAL_PEOPLE_DATA.is_recid != -1
]

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

streamlit.subheader(
    "TODO: Descriptive Statistics"
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
    ":bar_chart: " + translation("data_visualization.data_visualization_head")
)

streamlit.subheader(translation("data_visualization.correlation_matrix_label"))

correlation_columns = list(CRIMINAL_PEOPLE_DATA.columns)
correlation_columns.remove("decile_score")
correlation_columns.remove("c_days_from_compas")
correlation_columns.remove("num_r_cases")
correlation_columns.remove("r_days_from_arrest")

criminal_people_data_certain_columns = CRIMINAL_PEOPLE_DATA[
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
    criminal_people_data_certain_columns = criminal_people_data_certain_columns[
        criminal_people_data_certain_columns.sex == correlation_gender
    ]

criminal_people_data_correlations = criminal_people_data_certain_columns.corr()

streamlit.write("TODO: Pearson Correlation")

correlation_data = criminal_people_data_correlations.stack().reset_index().rename(
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
correlation_plot_text = base_correlation_plot.mark_text().encode(
    text="correlation_label",
    color=altair.condition(
        altair.datum.correlation > 0.5,
        altair.value("white"),
        altair.value("black")
    )
)

correlation_plot = base_correlation_plot.mark_rect().encode(
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

streamlit.altair_chart(
    CORRELATION_MATRIX.properties(
        width=800,
        height=800
    )
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
    CRIMINAL_PEOPLE_DATA[
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
    CRIMINAL_PEOPLE_DATA[
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

streamlit.altair_chart(
    boxplot_chart
)

streamlit.header(
    ":slot_machine: " + translation("data_classifier.header_classifier_training")
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
]

CATEGORIES_COLUMNS = [
    "sex",
    "race"
]

SELECTED_INPUT_DATA_FEATURES = streamlit.multiselect(
    translation("data_classifier.select_features_label"),
    options=INPUT_DATA_FEATURES + CATEGORIES_COLUMNS,
    default=INPUT_DATA_FEATURES
)

INPUT_DATA = CRIMINAL_PEOPLE_DATA[
    SELECTED_INPUT_DATA_FEATURES
]

SELECTED_CATEGORIES_COLUMNS = set(
    SELECTED_INPUT_DATA_FEATURES
) - set(INPUT_DATA_FEATURES)

INPUT_DATA = pandas.get_dummies(
    INPUT_DATA,
    columns=list(SELECTED_CATEGORIES_COLUMNS)
)

LABEL_DATA = CRIMINAL_PEOPLE_DATA[
    [
        "is_recid"
    ]
]

TRAIN_TEST_SPLIT_CODE_EXPANDER = streamlit.beta_expander(
    translation("data_classifier.train_test_split_code"),
    expanded=True
)

TRAIN_PERCENTAGE, TEST_PERCENTAGE = (75, 25)

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

    for index, target in enumerate(["is_not_recid", "is_recid"]):
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

CONFUSION_COLUMN.code(
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

CONFUSION_COLUMN.markdown(
    markdown_list(
        *[
            f"`{key}` / **{name}**: `{value}` / `{TEST_DATA_COUNT}` = "
            f"`{(value / TEST_DATA_COUNT):.2f}` ({description})"
            for key, name, description, value in [
                (
                    "tp",
                    translation("data_classifier.tp_name"),
                    translation("data_classifier.tp_description"),
                    tp
                ),
                (
                    "tn",
                    translation("data_classifier.tn_name"),
                    translation("data_classifier.tn_description"),
                    tn
                ),
                (
                    "fp",
                    translation("data_classifier.fp_name"),
                    translation("data_classifier.fp_description"),
                    fp
                ),
                (
                    "fn",
                    translation("data_classifier.fn_name"),
                    translation("data_classifier.fn_description"),
                    fn
                )
            ]
        ]
    )
)

METRICS_COLUMN.markdown(
    "#### " + translation("data_classifier.related_metrics")
)

# Accuracy
METRICS_COLUMN.markdown(
    "##### " + translation("data_classifier.accuracy")
)

METRICS_COLUMN.latex(
    "\\frac{TP + TN}{TP + TN + FP + FN} = " + f"{(tp + tn) / TEST_DATA_COUNT:.2f}"
)

# Precision
METRICS_COLUMN.markdown(
    "##### " + translation("data_classifier.precision")
)

METRICS_COLUMN.latex(
    "\\frac{TP}{TP + FP} = " + f"{tp / (tp + fp):.2f}"
)

# Recall
METRICS_COLUMN.markdown(
    "##### " + translation("data_classifier.recall")
)

METRICS_COLUMN.latex(
    "\\frac{TP}{TP + FN} = " + f"{tp / (tp + fn):.2f}"
)

streamlit.subheader("TODO: What happens if we add `race` to the input data?")

streamlit.header(
    "🧭 " + "TODO: Explanation of COMPAS"
)

streamlit.write(
    "TODO: Critical assessment of data insights"
)

streamlit.write(
    "TODO: Ethical view. "
    "Prevent misappropriation of system"
)

streamlit.write(
    "TODO: Use systems only where it can be beneficial. "
    "i.e. Target high recidivism risk people for more rehabilitation programmes"
)

streamlit.header(
    "👆 " +
    "TODO: COMPAS Interactive Threshold Choosing"
)

streamlit.markdown(
    "TODO: Create one visualization for all visualizations"
)

streamlit.header(
    ":clipboard: " +
    "TODO: Summary"
)

OUTLINE_COMMENTS = [
    "1",
    "2",
    "3",
    "4",
    "5"
]

OUTLINE_PREVIEW_ITEMS = [
    CRIMINAL_PEOPLE_DATA.head(n=3),
    CORRELATION_MATRIX.properties(width=400, height=300),
    "![scikit-learn](https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg)",
    "4",
    "5"
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
    "TODO: Streamlit und Quellen"
)
