
import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

API_URL = "https://flower-vision-api-sm13.onrender.com/predict"

FLOWER_EMOJIS = {
    "daisy": "🌼",
    "dandelion": "🌻",
    "roses": "🌹",
    "sunflowers": "🌻",
    "tulips": "🌷"
}

FLOWER_DESCRIPTIONS = {
    "daisy": "A cheerful flower known for its delicate petals and bright centre.",
    "dandelion": "A familiar golden flower known for its fluffy seed heads.",
    "roses": "A beautiful flower celebrated for its layered petals and rich colours.",
    "sunflowers": "A bright flower recognized by its large golden petals.",
    "tulips": "An elegant flower with distinctive cup-shaped petals."
}

st.set_page_config(
    page_title="FlowerVision | Flower Recognition",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

.stApp {
    background-color: #FBF8F5;
    color: #382D32;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #49343E;
}

.hero {
    text-align: center;
    padding: 2.5rem 1rem 2rem;
}

.hero-tag {
    color: #B66D83;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.5rem, 6vw, 4.5rem);
    color: #49343E;
    margin: 0.5rem 0;
}

.hero-subtitle {
    color: #85747A;
    font-size: 1.05rem;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.7;
}

.section-label {
    color: #B66D83;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.upload-card {
    background: #FFFFFF;
    border: 1px solid #F0E2E6;
    border-radius: 18px;
    padding: 1.5rem;
    box-shadow: 0 5px 25px rgba(90, 54, 69, 0.04);
}

.result-card {
    background: #FFFFFF;
    border: 1px solid #F0E2E6;
    border-radius: 18px;
    padding: 1.5rem;
    box-shadow: 0 5px 25px rgba(90, 54, 69, 0.04);
}

.prediction-name {
    font-family: 'Playfair Display', serif;
    color: #49343E;
    font-size: 2.2rem;
    font-weight: 600;
}

.confidence {
    color: #B66D83;
    font-size: 2.5rem;
    font-weight: 700;
}

.stButton > button {
    background-color: #B66D83;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 1.2rem;
    font-weight: 600;
    width: 100%;
}

.stButton > button:hover {
    background-color: #96546A;
    color: white;
    border: none;
}

hr {
    border-color: #F0E2E6;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="hero-tag">✿ Discover the beauty of nature ✿</div>
    <div class="hero-title">FlowerVision</div>
    <div class="hero-subtitle">
        A little intelligence, a little nature, and a whole lot of flowers.
        Upload a photograph and let AI identify the flower for you.
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-label">01 — Your photograph</div>',
                unsafe_allow_html=True)

    st.markdown("### Choose a flower")

    uploaded_file = st.file_uploader(
        "Upload a flower photograph",
        type=["jpg", "jpeg", "png"],
        help="Choose a clear photograph of a daisy, dandelion, rose, sunflower, or tulip."
    )

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        st.image(
            image,
            caption="Your uploaded photograph",
            use_container_width=True
        )

        st.caption(
            f"Image dimensions: {image.width} × {image.height} pixels"
        )

        identify_button = st.button(
            "✿ Identify this flower",
            use_container_width=True
        )

    else:
        st.info(
            "Upload a JPG or PNG photograph to get started."
        )
        identify_button = False


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

with right:
    st.markdown('<div class="section-label">02 — AI identification</div>',
                unsafe_allow_html=True)

    st.markdown("### Your results")

    if uploaded_file and identify_button:

        with st.spinner(
            "Looking at your flower... this may take a moment if the API is waking up."
        ):
            try:
                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            image_bytes,
                            uploaded_file.type
                        )
                    },
                    timeout=180
                )

                response.raise_for_status()
                result = response.json()

                predicted_class = result["predicted_class"]
                confidence = result["confidence"]
                probabilities = result["probabilities"]

                emoji = FLOWER_EMOJIS.get(
                    predicted_class,
                    "🌸"
                )

                description = FLOWER_DESCRIPTIONS.get(
                    predicted_class,
                    "Your flower has been identified."
                )

                st.success("Flower successfully identified!")

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div style="font-size:3rem;text-align:center;">
                            {emoji}
                        </div>
                        <div style="text-align:center;">
                            <div style="color:#B66D83;font-size:0.8rem;
                            font-weight:700;letter-spacing:2px;">
                                YOUR FLOWER IS
                            </div>
                            <div class="prediction-name">
                                {predicted_class.title()}
                            </div>
                            <div class="confidence">
                                {confidence:.2%}
                            </div>
                            <div style="color:#85747A;">
                                Prediction confidence
                            </div>
                            <p style="color:#85747A;margin-top:1rem;">
                                {description}
                            </p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("#### Class probabilities")
                st.caption(
                    "How strongly the model associates your image with each flower class."
                )

                sorted_probabilities = sorted(
                    probabilities.items(),
                    key=lambda item: item[1],
                    reverse=True
                )

                for flower, probability in sorted_probabilities:
                    st.markdown(
                        f"**{FLOWER_EMOJIS.get(flower, '🌸')} "
                        f"{flower.title()}**"
                    )

                    st.progress(
                        min(max(float(probability), 0.0), 1.0),
                        text=f"{float(probability):.2%}"
                    )

            except requests.exceptions.Timeout:
                st.error(
                    "The prediction request timed out. "
                    "The server may be waking up. Please try again."
                )

            except requests.exceptions.RequestException as error:
                st.error(
                    f"Could not connect to the FlowerVision API: {error}"
                )

            except Exception as error:
                st.error(
                    f"Something went wrong: {error}"
                )

    elif not uploaded_file:
        st.info(
            "Your flower identification results will appear here "
            "after you upload a photograph."
        )

    else:
        st.info(
            "Your photograph is ready! Click "
            "'Identify this flower' to get your prediction."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown("""
<div style="text-align:center;padding:1.5rem 0;color:#927F86;">
    <div style="font-family:'Playfair Display',serif;
    font-size:1.3rem;color:#49343E;">
        FlowerVision 🌸
    </div>
    <p style="font-size:0.85rem;">
        Powered by MobileNetV2 · ONNX Runtime · FastAPI
    </p>
    <p style="font-size:0.75rem;">
        Made with curiosity and a love for flowers.
    </p>
</div>
""", unsafe_allow_html=True)
