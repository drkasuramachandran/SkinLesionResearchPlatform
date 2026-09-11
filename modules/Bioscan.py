import streamlit as st
import numpy as np
import cv2
import tifffile
import matplotlib.pyplot as plt
from PIL import Image
import io


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BioColorScan",
    page_icon="🔬",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🔬 BioColorScan")

st.markdown(
    """
    **Biomedical Skin Image Analysis**

    Upload a skin image to perform image preprocessing,
    pixel-wise ITA calculation, statistical analysis, and
    visualization.
    """
)


# ============================================================
# FUNCTIONS
# ============================================================

def load_image(uploaded_file):
    """
    Load JPG, JPEG, PNG, BMP, TIF and TIFF images.
    Returns RGB NumPy array.
    """

    file_name = uploaded_file.name.lower()

    if file_name.endswith((".tif", ".tiff")):

        img = tifffile.imread(uploaded_file)

        # Handle grayscale TIFF
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Handle RGBA TIFF
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Convert to uint8 if necessary
        if img.dtype != np.uint8:

            img = cv2.normalize(
                img,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)

        return img

    else:

        image_pil = Image.open(uploaded_file).convert("RGB")

        return np.array(image_pil).astype(np.uint8)


def calculate_ita(image_rgb):
    """
    Calculate pixel-wise Individual Typology Angle (ITA).
    """

    # RGB → LAB
    lab = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2LAB
    ).astype(np.float32)

    # OpenCV LAB conversion
    L = lab[:, :, 0] * 100.0 / 255.0
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0

    epsilon = 1e-6

    ITA_map = np.degrees(
        np.arctan(
            (L - 50.0) /
            (b + epsilon)
        )
    )

    return L, a, b, ITA_map


def create_ita_color_map(ITA_map):

    ITA_display = cv2.normalize(
        ITA_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    ITA_color = cv2.applyColorMap(
        ITA_display,
        cv2.COLORMAP_VIRIDIS
    )

    ITA_color_rgb = cv2.cvtColor(
        ITA_color,
        cv2.COLOR_BGR2RGB
    )

    return ITA_display, ITA_color_rgb


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload skin image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "tif",
        "tiff"
    ]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image_rgb = load_image(uploaded_file)

    height, width = image_rgb.shape[:2]

    # --------------------------------------------------------
    # IMAGE INFORMATION
    # --------------------------------------------------------

    st.divider()

    st.subheader("Image Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Width",
            f"{width} px"
        )

    with col2:
        st.metric(
            "Height",
            f"{height} px"
        )

    with col3:
        st.metric(
            "Channels",
            image_rgb.shape[2]
        )

    with col4:
        st.metric(
            "Data Type",
            str(image_rgb.dtype)
        )

    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    st.divider()

    st.subheader("Original Image")

    st.image(
        image_rgb,
        use_container_width=True
    )

    # ========================================================
    # IMAGE PREPROCESSING
    # ========================================================

    st.divider()

    st.header("Image Preprocessing")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Hair Removal")

        remove_hair = st.checkbox(
            "Enable hair removal"
        )

    with col2:

        st.subheader("Illumination Correction")

        illumination_correction = st.checkbox(
            "Enable illumination correction"
        )

    # --------------------------------------------------------
    # PREPROCESSING STATUS
    # --------------------------------------------------------

    processed_image = image_rgb.copy()

    if remove_hair:

        st.info(
            "Hair-removal algorithm will be applied here."
        )

        # ----------------------------------------------------
        # PLACEHOLDER
        # Your hair removal algorithm can be inserted here.
        # ----------------------------------------------------

    if illumination_correction:

        st.info(
            "Illumination-correction algorithm will be "
            "applied here."
        )

        # ----------------------------------------------------
        # PLACEHOLDER
        # Your illumination correction algorithm can be
        # inserted here.
        # ----------------------------------------------------

    # --------------------------------------------------------
    # DISPLAY PROCESSED IMAGE
    # --------------------------------------------------------

    if remove_hair or illumination_correction:

        st.subheader("Preprocessed Image")

        st.image(
            processed_image,
            use_container_width=True
        )

    # ========================================================
    # ITA ANALYSIS
    # ========================================================

    st.divider()

    st.header("ITA Analysis")

    calculate_button = st.button(
        "🔬 Calculate Pixel-wise ITA",
        type="primary"
    )

    if calculate_button:

        # ----------------------------------------------------
        # CALCULATE ITA
        # ----------------------------------------------------

        with st.spinner("Calculating pixel-wise ITA..."):

            L, a, b, ITA_map = calculate_ita(
                processed_image
            )

        st.success(
            "Pixel-wise ITA calculated successfully."
        )

        # ====================================================
        # ITA STATISTICS
        # ====================================================

        st.subheader("ITA Statistics")

        ita_min = np.min(ITA_map)
        ita_max = np.max(ITA_map)
        ita_mean = np.mean(ITA_map)
        ita_median = np.median(ITA_map)
        ita_std = np.std(ITA_map)

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Minimum",
                f"{ita_min:.2f}°"
            )

        with col2:
            st.metric(
                "Maximum",
                f"{ita_max:.2f}°"
            )

        with col3:
            st.metric(
                "Mean",
                f"{ita_mean:.2f}°"
            )

        with col4:
            st.metric(
                "Median",
                f"{ita_median:.2f}°"
            )

        with col5:
            st.metric(
                "Std. Dev.",
                f"{ita_std:.2f}°"
            )

        # ====================================================
        # ITA VISUALIZATION
        # ====================================================

        st.divider()

        st.header("ITA Visualization")

        # ----------------------------------------------------
        # Create false-colour map
        # ----------------------------------------------------

        ITA_display, ITA_color_rgb = create_ita_color_map(
            ITA_map
        )

        # ----------------------------------------------------
        # Two-column display
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Pixel-wise ITA Map")

            fig1, ax1 = plt.subplots(
                figsize=(7, 6)
            )

            im1 = ax1.imshow(
                ITA_map,
                cmap="viridis"
            )

            ax1.set_title(
                "Pixel-wise ITA Map"
            )

            ax1.axis("off")

            fig1.colorbar(
                im1,
                ax=ax1,
                label="ITA (Degrees)"
            )

            st.pyplot(
                fig1,
                use_container_width=True
            )

            plt.close(fig1)

        with col2:

            st.subheader("False Colour ITA Map")

            fig2, ax2 = plt.subplots(
                figsize=(7, 6)
            )

            ax2.imshow(
                ITA_color_rgb
            )

            ax2.set_title(
                "False Colour ITA Map"
            )

            ax2.axis("off")

            st.pyplot(
                fig2,
                use_container_width=True
            )

            plt.close(fig2)

        # ====================================================
        # ITA HISTOGRAM
        # ====================================================

        st.divider()

        st.subheader("ITA Distribution")

        fig3, ax3 = plt.subplots(
            figsize=(10, 5)
        )

        ax3.hist(
            ITA_map.flatten(),
            bins=100,
            edgecolor="black"
        )

        ax3.set_xlabel(
            "ITA (Degrees)"
        )

        ax3.set_ylabel(
            "Number of Pixels"
        )

        ax3.set_title(
            "Histogram of Pixel-wise ITA"
        )

        ax3.grid(True)

        st.pyplot(
            fig3,
            use_container_width=True
        )

        plt.close(fig3)

        # ====================================================
        # SMOOTH ITA MAP
        # ====================================================

        st.divider()

        st.subheader("Smoothed ITA Map")

        ITA_smooth = cv2.GaussianBlur(
            ITA_map,
            (11, 11),
            0
        )

        fig4, ax4 = plt.subplots(
            figsize=(8, 7)
        )

        im4 = ax4.imshow(
            ITA_smooth,
            cmap="viridis"
        )

        ax4.set_title(
            "Smoothed Pixel-wise ITA Map"
        )

        ax4.axis("off")

        fig4.colorbar(
            im4,
            ax=ax4,
            label="ITA (Degrees)"
        )

        st.pyplot(
            fig4,
            use_container_width=True
        )

        plt.close(fig4)

        # ====================================================
        # 3D ITA SURFACE
        # ====================================================

        st.divider()

        st.subheader("3D Pixel-wise ITA Surface")

        # Downsample large images for faster rendering
        max_dimension = 300

        scale = min(
            1.0,
            max_dimension /
            max(ITA_map.shape)
        )

        if scale < 1.0:

            small_ita = cv2.resize(
                ITA_map,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA
            )

        else:

            small_ita = ITA_map

        rows, cols = small_ita.shape

        X = np.arange(cols)
        Y = np.arange(rows)

        X, Y = np.meshgrid(X, Y)

        fig5 = plt.figure(
            figsize=(12, 8)
        )

        ax5 = fig5.add_subplot(
            111,
            projection="3d"
        )

        surface = ax5.plot_surface(
            X,
            Y,
            small_ita,
            cmap="viridis",
            linewidth=0,
            antialiased=True
        )

        ax5.set_title(
            "3D Pixel-wise ITA Surface"
        )

        ax5.set_xlabel(
            "X Pixel"
        )

        ax5.set_ylabel(
            "Y Pixel"
        )

        ax5.set_zlabel(
            "ITA (Degrees)"
        )

        fig5.colorbar(
            surface,
            shrink=0.6,
            label="ITA"
        )

        st.pyplot(
            fig5,
            use_container_width=True
        )

        plt.close(fig5)

        # ====================================================
        # DOWNLOAD RESULTS
        # ====================================================

        st.divider()

        st.header("Download Results")

        # ----------------------------------------------------
        # ITA NumPy file
        # ----------------------------------------------------

        ita_npy_buffer = io.BytesIO()

        np.save(
            ita_npy_buffer,
            ITA_map
        )

        ita_npy_buffer.seek(0)

        # ----------------------------------------------------
        # ITA CSV
        # ----------------------------------------------------

        ita_csv = io.StringIO()

        np.savetxt(
            ita_csv,
            ITA_map,
            delimiter=",",
            fmt="%.4f"
        )

        # ----------------------------------------------------
        # ITA PNG
        # ----------------------------------------------------

        ita_png_buffer = io.BytesIO()

        Image.fromarray(
            ITA_color_rgb
        ).save(
            ita_png_buffer,
            format="PNG"
        )

        ita_png_buffer.seek(0)

        col1, col2, col3 = st.columns(3)

        with col1:

            st.download_button(
                "⬇ Download ITA Map (.npy)",
                data=ita_npy_buffer,
                file_name="ITA_Map.npy",
                mime="application/octet-stream"
            )

        with col2:

            st.download_button(
                "⬇ Download ITA Data (.csv)",
                data=ita_csv.getvalue(),
                file_name="ITA_Map.csv",
                mime="text/csv"
            )

        with col3:

            st.download_button(
                "⬇ Download ITA Colour Map (.png)",
                data=ita_png_buffer,
                file_name="ITA_Color_Map.png",
                mime="image/png"
            )

        # ====================================================
        # LAB INFORMATION
        # ====================================================

        st.divider()

        st.subheader("CIELAB Information")

        lab_col1, lab_col2, lab_col3 = st.columns(3)

        with lab_col1:

            st.metric(
                "Mean L*",
                f"{np.mean(L):.2f}"
            )

        with lab_col2:

            st.metric(
                "Mean a*",
                f"{np.mean(a):.2f}"
            )

        with lab_col3:

            st.metric(
                "Mean b*",
                f"{np.mean(b):.2f}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "BioColorScan | Biomedical Skin Image Analysis"
)

