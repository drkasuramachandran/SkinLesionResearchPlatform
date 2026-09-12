# ============================================================
# SKIN LESION RESEARCH PLATFORM
# Complete Streamlit Application — Combined Version
# ============================================================

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd


# ============================================================
# EXISTING MODULES
# ============================================================

from modules.cnn import (
    load_cnn_model,
    predict_image
)

from modules.abcde import (
    analyze_abcd
)

from modules.cielab_ita import (
    analyze_cielab_ita
)


# ============================================================
# EXPERIMENT MODULE
# ============================================================

from modules.experiments import (
    analyze_experiment_a,
    analyze_experiment_b,
    analyze_experiment_c,
    analyze_experiment_d,
    analyze_experiment_e
)


# ============================================================
# ROI ITA MODULE
# ============================================================

try:

    from modules.roi_ita import (
        analyze_roi_pixelwise_ita,
        analyze_three_region_ita,
        create_region_ita_map,
        create_ita_display
    )

    ROI_ITA_AVAILABLE = True

except ImportError as e:

    ROI_ITA_AVAILABLE = False
    ROI_ITA_IMPORT_ERROR = str(e)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Skin Lesion Research Platform",
    page_icon="🔬",
    layout="wide"
)


# ============================================================
# VISUAL THEME
# ============================================================

st.markdown(
    """
    <style>
    /* Main application background */
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(20, 115, 125, 0.18), transparent 28%),
            radial-gradient(circle at 85% 80%, rgba(15, 90, 105, 0.16), transparent 30%),
            linear-gradient(135deg, #07181d 0%, #0a2228 48%, #07161b 100%);
        color: #e8f4f5;
    }

    /* Main content */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, #061419 0%, #0a2026 100%);
        border-right: 1px solid rgba(92, 210, 215, 0.18);
    }

    [data-testid="stSidebar"] * {
        color: #dcebed;
    }

    /* Main headings */
    h1, h2, h3 {
        color: #f1fbfc !important;
        letter-spacing: 0.2px;
    }

    h1 {
        text-shadow: 0 0 18px rgba(80, 210, 215, 0.16);
    }

    /* Normal text */
    p, li, label, .stMarkdown {
        color: #d5e5e7;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: rgba(8, 31, 37, 0.72);
        border: 1px solid rgba(83, 193, 199, 0.20);
        border-radius: 12px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.16);
    }

    [data-testid="stExpander"] summary {
        color: #e9f7f8;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(12, 42, 49, 0.72);
        border: 1px solid rgba(83, 193, 199, 0.22);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.14);
    }

    [data-testid="stMetricLabel"] {
        color: #a9c7ca !important;
    }

    [data-testid="stMetricValue"] {
        color: #f2ffff !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #167f87, #20a5a9);
        color: white;
        border: 1px solid rgba(150, 245, 245, 0.22);
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1b9299, #29b7ba);
        border-color: rgba(180, 255, 255, 0.45);
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(20, 170, 175, 0.18);
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(8, 31, 37, 0.58);
        border: 1px dashed rgba(92, 210, 215, 0.35);
        border-radius: 12px;
        padding: 0.35rem;
    }

    /* Info / success / warning / error boxes */
    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(83, 193, 199, 0.18);
        border-radius: 10px;
        overflow: hidden;
    }

    /* Progress bar */
    [data-testid="stProgress"] > div > div > div {
        background: linear-gradient(90deg, #167f87, #35c4c4);
    }

    /* Dividers */
    hr {
        border-color: rgba(92, 210, 215, 0.16);
    }

    /* Links */
    a {
        color: #67d9dc !important;
    }

    /* Captions */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #91b1b5 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# ============================================================
# TITLE
# ============================================================

st.title(
    "🔬 Skin Lesion Research Platform"
)

st.markdown(
    """
    **Research-oriented platform for skin lesion image analysis**

    The platform combines image processing, CNN classification,
    ABCD analysis, CIELAB/ITA analysis, ImageJ ROI analysis,
    and preprocessing experiments.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Research Modules"
)

st.sidebar.markdown(
    """
    ### Available modules

    - Image Processing
    - CNN Classification
    - ABCD Analysis
    - CIELAB / ITA
    - Baseline ROI ITA
    - Three-Region ImageJ ROI ITA
    - Experiment A
    - Experiment B
    - Experiment C
    - Experiment D
    - Experiment E
    - Image Evolution
    - Research Summary
    """
)

st.sidebar.divider()

st.sidebar.info(
    """
    **Research use only**

    The outputs are intended for research,
    educational and methodological evaluation.
    They are not intended for clinical diagnosis.
    """
)


# ============================================================
# CNN MODEL
# ============================================================

@st.cache_resource
def get_cnn_model():

    return load_cnn_model()


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.header(
    "1. Image Upload"
)

uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "tif",
        "tiff"
    ]
)


# ============================================================
# NO IMAGE
# ============================================================

if uploaded_file is None:

    st.info(
        "Please upload a skin lesion image to begin the analysis."
    )

    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    image = Image.open(
        uploaded_file
    ).convert(
        "RGB"
    )

except Exception as e:

    st.error(
        f"Unable to read the image: {e}"
    )

    st.stop()


# ============================================================
# IMAGE ARRAYS
# ============================================================

image_rgb = np.array(
    image
)

image_bgr = cv2.cvtColor(
    image_rgb,
    cv2.COLOR_RGB2BGR
)


# ============================================================
# IMAGE INFORMATION
# ============================================================

st.subheader(
    "Uploaded Image"
)

col1, col2 = st.columns(
    [2, 1]
)

with col1:

    st.image(
        image,
        caption="Original Image",
        use_container_width=True
    )

with col2:

    st.markdown(
        "### Image Information"
    )

    st.write(
        f"**Filename:** {uploaded_file.name}"
    )

    st.write(
        f"**Width:** {image.width} pixels"
    )

    st.write(
        f"**Height:** {image.height} pixels"
    )

    st.write(
        f"**Color mode:** {image.mode}"
    )


# ============================================================
# SECTION 2
# IMAGE PROCESSING
# ============================================================

st.header(
    "2. Image Processing"
)

with st.expander(
    "View image processing",
    expanded=False
):

    st.image(
        image,
        caption="Original RGB Image",
        use_container_width=True
    )

    st.write(
        """
        The uploaded image is converted into RGB and BGR
        representations for subsequent image-processing
        operations.
        """
    )


# ============================================================
# SECTION 3
# CNN CLASSIFICATION
# ============================================================

st.header(
    "3. CNN Classification"
)

with st.expander(
    "CNN Classification",
    expanded=False
):

    try:

        model = get_cnn_model()

        probability, prediction = predict_image(
            model,
            image
        )

        malignant_probability = (
            probability * 100
        )

        benign_probability = (
            (1 - probability) * 100
        )

        col1, col2, col3 = st.columns(
            3
        )

        with col1:

            st.metric(
                "Prediction",
                prediction
            )

        with col2:

            st.metric(
                "Malignant Probability",
                f"{malignant_probability:.2f}%"
            )

        with col3:

            st.metric(
                "Benign Probability",
                f"{benign_probability:.2f}%"
            )

        st.progress(
            probability
        )

        st.caption(
            """
            The CNN output is a research classification result
            and should not be interpreted as a clinical diagnosis.
            """
        )

    except Exception as e:

        st.error(
            f"CNN analysis failed: {e}"
        )


# ============================================================
# SECTION 4
# ABCD ANALYSIS
# ============================================================

st.header(
    "4. ABCD Analysis"
)

with st.expander(
    "ABCD Analysis",
    expanded=False
):

    try:

        abcd_results, lesion_mask = (
            analyze_abcd(
                image_bgr
            )
        )

        col1, col2, col3, col4 = st.columns(
            4
        )

        with col1:

            st.metric(
                "Asymmetry",
                f"{abcd_results['Asymmetry Index']:.3f}"
            )

        with col2:

            st.metric(
                "Circularity",
                f"{abcd_results['Circularity']:.3f}"
            )

        with col3:

            st.metric(
                "Color Variation",
                f"{abcd_results['Color Variation']:.2f}"
            )

        with col4:

            st.metric(
                "Diameter",
                f"{abcd_results['Diameter (mm)']:.2f} mm"
            )

        st.markdown(
            "### ABCD Score"
        )

        st.metric(
            "Score",
            f"{abcd_results['ABCD Score']} / "
            f"{abcd_results['Maximum Score']}"
        )

        mask_display = (
            lesion_mask > 0
        ).astype(
            np.uint8
        ) * 255

        st.image(
            mask_display,
            caption="Estimated Lesion Mask",
            use_container_width=True
        )

        st.caption(
            """
            Diameter currently uses the default pixel calibration
            defined in the ABCD module. Actual physical measurements
            require image calibration.
            """
        )

    except Exception as e:

        st.error(
            f"ABCD analysis failed: {e}"
        )


# ============================================================
# SECTION 5
# CIELAB / ITA
# ============================================================

st.header(
    "5. CIELAB / ITA Analysis"
)

with st.expander(
    "CIELAB / ITA Analysis",
    expanded=False
):

    try:

        ita_results = (
            analyze_cielab_ita(
                image_bgr
            )
        )

        results = ita_results[
            "results"
        ]

        ita_map = ita_results[
            "ita_map"
        ]

        ita_display = ita_results[
            "ita_display"
        ]

        ita_color = ita_results[
            "ita_color"
        ]

        ita_smooth = ita_results[
            "ita_smooth"
        ]

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        col1, col2, col3, col4, col5 = st.columns(
            5
        )

        with col1:

            st.metric(
                "Minimum ITA",
                f"{results['Minimum ITA']:.2f}"
            )

        with col2:

            st.metric(
                "Maximum ITA",
                f"{results['Maximum ITA']:.2f}"
            )

        with col3:

            st.metric(
                "Mean ITA",
                f"{results['Mean ITA']:.2f}"
            )

        with col4:

            st.metric(
                "Median ITA",
                f"{results['Median ITA']:.2f}"
            )

        with col5:

            st.metric(
                "Std. Dev.",
                f"{results['Standard Deviation']:.2f}"
            )

        # ----------------------------------------------------
        # ITA map
        # ----------------------------------------------------

        st.markdown(
            "### ITA Map"
        )

        st.image(
            ita_color,
            caption="ITA Jet Colormap",
            use_container_width=True
        )

        # ----------------------------------------------------
        # Smoothed ITA
        # ----------------------------------------------------

        st.markdown(
            "### Smoothed ITA Map"
        )

        smooth_display = cv2.normalize(
            ita_smooth,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(
            np.uint8
        )

        st.image(
            smooth_display,
            caption="Smoothed ITA",
            use_container_width=True
        )

        # ----------------------------------------------------
        # Histogram
        # ----------------------------------------------------

        st.markdown(
            "### ITA Histogram"
        )

        histogram_values = (
            ita_map[
                np.isfinite(ita_map)
            ]
            .flatten()
        )

        if len(histogram_values) > 0:

            histogram_df = pd.DataFrame(
                {
                    "ITA": histogram_values
                }
            )

            st.bar_chart(
                histogram_df[
                    "ITA"
                ].value_counts(
                    bins=50
                ).sort_index()
            )

    except Exception as e:

        st.error(
            f"CIELAB / ITA analysis failed: {e}"
        )


# ============================================================
# SECTION 6
# IMAGEJ ROI ANALYSIS
# ============================================================

st.header(
    "6. ImageJ ROI Analysis"
)

st.markdown(
    """
    Upload the ImageJ ROI ZIP file corresponding to the uploaded
    lesion image.

    The ZIP should contain the ImageJ `.roi` files used to define
    ROI 1, ROI 2 and ROI 3.
    """
)

roi_zip = st.file_uploader(
    "Upload ImageJ ROI ZIP file",
    type=["zip"],
    key="roi_zip"
)


# ============================================================
# ROI ANALYSIS
# ============================================================

if roi_zip is not None:

    st.success(
        f"ROI file uploaded: {roi_zip.name}"
    )

    # ========================================================
    # 6.1 BASELINE ROI ITA
    # ========================================================

    st.subheader(
        "6.1 Baseline ROI ITA"
    )

    if ROI_ITA_AVAILABLE:

        with st.expander(
            "Baseline pixel-wise ROI ITA",
            expanded=False
        ):

            try:

                baseline_results = (
                    analyze_roi_pixelwise_ita(
                        image_bgr,
                        roi_zip
                    )
                )

                if isinstance(baseline_results, dict):

                    if "results" in baseline_results:

                        baseline_table = pd.DataFrame(
                            baseline_results["results"]
                        )

                    else:

                        baseline_table = pd.DataFrame(
                            baseline_results
                        )

                else:

                    baseline_table = pd.DataFrame(
                        baseline_results
                    )

                st.dataframe(
                    baseline_table,
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"Baseline ROI ITA analysis failed: {e}"
                )

    # ========================================================
    # 6.2 THREE-REGION IMAGEJ ROI ITA
    # ========================================================

    st.subheader(
        "6.2 Three-Region ImageJ ROI Pixel-wise ITA Analysis"
    )

    st.markdown(
        """
        This analysis uses three nested ImageJ ROIs.

        **Region definitions**

        - **ROI 1:** Lesion
        - **ROI 2 − ROI 1:** Surrounding healthy skin
        - **ROI 3 − ROI 2:** Outer healthy skin

        ROI subtraction is performed at the pixel-mask level.
        ITA is then calculated independently for each physical region.
        """
    )

    if ROI_ITA_AVAILABLE:

        with st.expander(
            "Three-Region ROI ITA Analysis",
            expanded=False
        ):

            try:

                roi_output = analyze_three_region_ita(
                    image_bgr,
                    roi_zip
                )

                results = roi_output["results"]
                derived_masks = roi_output["derived_masks"]
                region_names = roi_output["region_names"]

                # ------------------------------------------------
                # REGION SUMMARY
                # ------------------------------------------------

                st.markdown(
                    "### Three-Region Analysis"
                )

                result_table = []

                for row in results:

                    result_table.append({

                        "Region":
                            row["Region"],

                        "Meaning":
                            row["Meaning"],

                        "Valid Pixels":
                            row["Valid Pixels"],

                        "Min ITA (°)":
                            (
                                f"{row['Minimum ITA']:.2f}"
                                if np.isfinite(
                                    row["Minimum ITA"]
                                )
                                else "N/A"
                            ),

                        "Max ITA (°)":
                            (
                                f"{row['Maximum ITA']:.2f}"
                                if np.isfinite(
                                    row["Maximum ITA"]
                                )
                                else "N/A"
                            ),

                        "Mean ITA (°)":
                            (
                                f"{row['Mean ITA']:.2f}"
                                if np.isfinite(
                                    row["Mean ITA"]
                                )
                                else "N/A"
                            ),

                        "Median ITA (°)":
                            (
                                f"{row['Median ITA']:.2f}"
                                if np.isfinite(
                                    row["Median ITA"]
                                )
                                else "N/A"
                            ),

                        "Std. Dev. (°)":
                            (
                                f"{row['Standard Deviation']:.2f}"
                                if np.isfinite(
                                    row["Standard Deviation"]
                                )
                                else "N/A"
                            )
                    })

                st.dataframe(
                    result_table,
                    use_container_width=True,
                    hide_index=True
                )

                # ------------------------------------------------
                # DERIVED REGION MASKS
                # ------------------------------------------------

                st.markdown(
                    "### Derived Analysis Regions"
                )

                st.caption(
                    "These masks show the actual pixel regions "
                    "used for ITA calculation."
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.image(
                        derived_masks[0],
                        caption="ROI 1 — Lesion",
                        clamp=True,
                        use_container_width=True
                    )

                with col2:

                    st.image(
                        derived_masks[1],
                        caption="ROI 2 − ROI 1 — Surrounding Healthy Skin",
                        clamp=True,
                        use_container_width=True
                    )

                with col3:

                    st.image(
                        derived_masks[2],
                        caption="ROI 3 − ROI 2 — Outer Healthy Skin",
                        clamp=True,
                        use_container_width=True
                    )

                # ------------------------------------------------
                # REGION-WISE ITA MAPS
                # ------------------------------------------------

                st.markdown(
                    "### Region-wise ITA Maps"
                )

                for index in range(3):

                    ita_region_map = create_region_ita_map(
                        image_bgr,
                        derived_masks[index]
                    )

                    ita_region_display = create_ita_display(
                        ita_region_map
                    )

                    st.image(
                        ita_region_display,
                        caption=region_names[index],
                        use_container_width=True
                    )

                # ------------------------------------------------
                # ITA DIFFERENCES
                # ------------------------------------------------

                st.markdown(
                    "### Mean ITA Differences"
                )

                mean_ita = [
                    row["Mean ITA"]
                    for row in results
                ]

                if all(
                    np.isfinite(value)
                    for value in mean_ita
                ):

                    delta_lesion_surrounding = (
                        mean_ita[0] -
                        mean_ita[1]
                    )

                    delta_surrounding_outer = (
                        mean_ita[1] -
                        mean_ita[2]
                    )

                    delta_lesion_outer = (
                        mean_ita[0] -
                        mean_ita[2]
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Lesion − Surrounding",
                            f"{delta_lesion_surrounding:.2f}°"
                        )

                    with col2:

                        st.metric(
                            "Surrounding − Outer",
                            f"{delta_surrounding_outer:.2f}°"
                        )

                    with col3:

                        st.metric(
                            "Lesion − Outer",
                            f"{delta_lesion_outer:.2f}°"
                        )

                # ------------------------------------------------
                # RESEARCH NOTE
                # ------------------------------------------------

                st.info(
                    "The three-region analysis uses the original "
                    "ImageJ ROIs to construct physical pixel regions. "
                    "ROI 1 represents the lesion, ROI 2 − ROI 1 "
                    "represents surrounding healthy skin, and "
                    "ROI 3 − ROI 2 represents outer healthy skin. "
                    "ITA is calculated pixel-wise using "
                    "atan2(L* − 50, b*)."
                )

            except Exception as e:

                st.error(
                    f"Three-region ImageJ ROI ITA analysis failed: {e}"
                )

    else:

        st.warning(
            """
            `modules.roi_ita` could not be imported.

            Check that `modules/roi_ita.py` is inside the `modules`
            folder and install its dependency with:

            `python -m pip install roifile`

            Import error: `{ROI_ITA_IMPORT_ERROR}`
            """
        )

else:

    st.info(
        "Upload the corresponding ImageJ ROI ZIP file to perform "
        "the baseline and three-region ROI ITA analyses, and to "
        "enable Experiments A-E."
    )


# ============================================================
# SECTION 7
# EXPERIMENTS A-E
# ============================================================

st.header(
    "7. Experiments A-E"
)

if roi_zip is None:

    st.info(
        """
        Upload the ImageJ ROI ZIP file above to enable
        Experiments A-E.
        """
    )

else:

    st.markdown(
        """
        The following experiments use the uploaded ImageJ ROIs.
        ROI 1, ROI 2 and ROI 3 are processed independently.

        **ROI pixel subtraction is not performed in Experiments A-E.**
        """
    )

    # ========================================================
    # EXPERIMENT A
    # ========================================================

    with st.expander(
        "Experiment A — ROI-only Shades of Gray → CIELAB → ITA",
        expanded=False
    ):

        st.markdown(
            """
            **Method**

            ImageJ ROI → ROI-only Shades of Gray (p=6)
            → CIELAB → mean L*, a*, b* → ITA

            Pixel-wise ITA statistics are also calculated.
            """
        )

        run_a = st.button(
            "Run Experiment A",
            key="run_experiment_a"
        )

        if run_a:

            try:

                with st.spinner(
                    "Running Experiment A..."
                ):

                    result_a = (
                        analyze_experiment_a(
                            image_bgr,
                            roi_zip,
                            p=6
                        )
                    )

                st.success(
                    "Experiment A completed."
                )

                df_a = pd.DataFrame(
                    result_a[
                        "results"
                    ]
                )

                st.dataframe(
                    df_a,
                    use_container_width=True
                )

                st.markdown(
                    "### Experiment A ITA"
                )

                if "ITA" in df_a.columns:

                    chart_a = df_a[
                        [
                            "Region",
                            "ITA"
                        ]
                    ].set_index(
                        "Region"
                    )

                    st.bar_chart(
                        chart_a
                    )

            except Exception as e:

                st.error(
                    f"Experiment A failed: {e}"
                )


    # ========================================================
    # EXPERIMENT B
    # ========================================================

    with st.expander(
        "Experiment B — Bounding-box Shades of Gray → CIELAB → ITA",
        expanded=False
    ):

        st.markdown(
            """
            **Method**

            ImageJ ROI → ROI bounding-box crop
            → Shades of Gray (p=6) using the entire crop
            → CIELAB → mean L*, a*, b* → ITA
            """
        )

        run_b = st.button(
            "Run Experiment B",
            key="run_experiment_b"
        )

        if run_b:

            try:

                with st.spinner(
                    "Running Experiment B..."
                ):

                    result_b = (
                        analyze_experiment_b(
                            image_bgr,
                            roi_zip,
                            p=6
                        )
                    )

                st.success(
                    "Experiment B completed."
                )

                df_b = pd.DataFrame(
                    result_b[
                        "results"
                    ]
                )

                st.dataframe(
                    df_b,
                    use_container_width=True
                )

                st.markdown(
                    "### Experiment B ITA"
                )

                if "ITA" in df_b.columns:

                    chart_b = df_b[
                        [
                            "Region",
                            "ITA"
                        ]
                    ].set_index(
                        "Region"
                    )

                    st.bar_chart(
                        chart_b
                    )

            except Exception as e:

                st.error(
                    f"Experiment B failed: {e}"
                )


    # ========================================================
    # EXPERIMENT C
    # ========================================================

    with st.expander(
        "Experiment C — Illumination correction + L* stretching",
        expanded=False
    ):

        st.markdown(
            """
            **Method 1**

            ROI-only Shades of Gray (p=6)
            → CIELAB → mean L*, a*, b* → ITA

            **Method 2**

            ROI-only Shades of Gray (p=6)
            → L* stretching → CIELAB
            → mean L*, a*, b* → ITA
            """
        )

        run_c = st.button(
            "Run Experiment C",
            key="run_experiment_c"
        )

        if run_c:

            try:

                with st.spinner(
                    "Running Experiment C..."
                ):

                    result_c = (
                        analyze_experiment_c(
                            image_bgr,
                            roi_zip,
                            p=6
                        )
                    )

                st.success(
                    "Experiment C completed."
                )

                df_c = pd.DataFrame(
                    result_c[
                        "results"
                    ]
                )

                st.dataframe(
                    df_c,
                    use_container_width=True
                )

                # --------------------------------------------
                # Method comparison
                # --------------------------------------------

                if (
                    "Method 1 ITA" in df_c.columns
                    and
                    "Method 2 ITA" in df_c.columns
                ):

                    st.markdown(
                        "### Method 1 vs Method 2"
                    )

                    comparison_c = df_c[
                        [
                            "Region",
                            "Method 1 ITA",
                            "Method 2 ITA"
                        ]
                    ].set_index(
                        "Region"
                    )

                    st.bar_chart(
                        comparison_c
                    )

            except Exception as e:

                st.error(
                    f"Experiment C failed: {e}"
                )


    # ========================================================
    # EXPERIMENT D
    # ========================================================

    with st.expander(
        "Experiment D — Hair removal → CIELAB → ITA",
        expanded=False
    ):

        st.markdown(
            """
            **Method**

            ImageJ ROI → hair removal
            → CIELAB → mean L*, a*, b* → ITA

            Hair removal is performed independently for each ROI.
            """
        )

        run_d = st.button(
            "Run Experiment D",
            key="run_experiment_d"
        )

        if run_d:

            try:

                with st.spinner(
                    "Running Experiment D..."
                ):

                    result_d = (
                        analyze_experiment_d(
                            image_bgr,
                            roi_zip
                        )
                    )

                st.success(
                    "Experiment D completed."
                )

                df_d = pd.DataFrame(
                    result_d[
                        "results"
                    ]
                )

                st.dataframe(
                    df_d,
                    use_container_width=True
                )

                if "ITA" in df_d.columns:

                    st.markdown(
                        "### Experiment D ITA"
                    )

                    chart_d = df_d[
                        [
                            "Region",
                            "ITA"
                        ]
                    ].set_index(
                        "Region"
                    )

                    st.bar_chart(
                        chart_d
                    )

            except Exception as e:

                st.error(
                    f"Experiment D failed: {e}"
                )


    # ========================================================
    # EXPERIMENT E
    # ========================================================

    with st.expander(
        "Experiment E — Illumination correction → Hair removal",
        expanded=False
    ):

        st.markdown(
            """
            **Step 1**

            ROI-only illumination correction
            → CIELAB → ITA

            **Step 2**

            Hair removal after illumination correction
            → CIELAB → ITA

            The two ITA measurements are retained separately.
            """
        )

        run_e = st.button(
            "Run Experiment E",
            key="run_experiment_e"
        )

        if run_e:

            try:

                with st.spinner(
                    "Running Experiment E..."
                ):

                    result_e = (
                        analyze_experiment_e(
                            image_bgr,
                            roi_zip,
                            p=6
                        )
                    )

                st.success(
                    "Experiment E completed."
                )

                df_e = pd.DataFrame(
                    result_e[
                        "results"
                    ]
                )

                st.dataframe(
                    df_e,
                    use_container_width=True
                )

                if (
                    "Illumination ITA" in df_e.columns
                    and
                    "Hair Removal ITA" in df_e.columns
                ):

                    st.markdown(
                        "### Illumination correction vs Hair removal"
                    )

                    comparison_e = df_e[
                        [
                            "Region",
                            "Illumination ITA",
                            "Hair Removal ITA"
                        ]
                    ].set_index(
                        "Region"
                    )

                    st.bar_chart(
                        comparison_e
                    )

            except Exception as e:

                st.error(
                    f"Experiment E failed: {e}"
                )


# ============================================================
# SECTION 8
# IMAGE EVOLUTION
# ============================================================

st.header(
    "8. Image Evolution"
)

with st.expander(
    "Longitudinal lesion comparison",
    expanded=False
):

    st.info(
        """
        Image Evolution will compare two images of the same lesion
        acquired at different time points.
        """
    )

    st.markdown(
        """
        Planned parameters:

        - Diameter
        - Asymmetry
        - Circularity
        - Color variation
        - Mean ITA
        - Median ITA
        - ΔITA
        - CNN malignant probability

        Image registration and normalization should be performed
        before quantitative comparison.
        """
    )

    evolution_old = st.file_uploader(
        "Upload earlier lesion image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "tif",
            "tiff"
        ],
        key="evolution_old"
    )

    evolution_new = st.file_uploader(
        "Upload later lesion image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "tif",
            "tiff"
        ],
        key="evolution_new"
    )

    if (
        evolution_old is not None
        and
        evolution_new is not None
    ):

        st.success(
            "Both images uploaded. Evolution analysis module can be connected next."
        )


# ============================================================
# SECTION 9
# RESEARCH SUMMARY
# ============================================================

st.header(
    "9. Research Summary"
)

with st.expander(
    "Research Summary",
    expanded=False
):

    st.info(
        """
        The Research Summary module will combine the results from
        the individual analysis modules into one research-oriented
        summary.
        """
    )

    st.markdown(
        """
        ### Planned summary

        **CNN**
        - Prediction
        - Malignant probability
        - Benign probability

        **ABCD**
        - Asymmetry
        - Circularity
        - Color variation
        - Diameter
        - ABCD score

        **CIELAB / ITA**
        - Minimum ITA
        - Maximum ITA
        - Mean ITA
        - Median ITA
        - Standard deviation

        **ROI Analysis**
        - ROI 1
        - ROI 2
        - ROI 3
        - Derived regional differences

        **Experiments**
        - A
        - B
        - C
        - D
        - E

        **Evolution**
        - Longitudinal changes

        **Export**
        - CSV
        - Excel
        - Research report
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    """
    Skin Lesion Research Platform | Research and educational use only
    """
)
