import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


st.set_page_config(
    page_title="Disease Prediction",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATASET_SOURCE = {
    "name": "Disease Prediction Using Machine Learning",
    "origin_url": "https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning",
    "train_url": "https://raw.githubusercontent.com/anujdutt9/Disease-Prediction-from-Symptoms/master/dataset/training_data.csv",
    "test_url": "https://raw.githubusercontent.com/anujdutt9/Disease-Prediction-from-Symptoms/master/dataset/test_data.csv",
}

DISEASE_INFO = {
    "Flu": "Often linked with fever, fatigue, body pain, chills, and cough.",
    "Common Cold": "Usually includes runny nose, sneezing, sore throat, and mild cough.",
    "COVID-19": "Can include fever, cough, fatigue, breathlessness, and loss of smell.",
    "Pneumonia": "Often shows cough, fever, chest pain, and shortness of breath.",
    "Asthma": "Common signs include wheezing, chest tightness, and shortness of breath.",
    "Bronchitis": "Often includes cough, fatigue, wheezing, and chest discomfort.",
    "Food Poisoning": "Typical pattern is nausea, vomiting, diarrhea, and abdominal pain.",
    "Gastritis": "Usually involves nausea, bloating, abdominal pain, and vomiting.",
    "Migraine": "Frequently includes headache, nausea, dizziness, and blurred vision.",
    "Diabetes": "Can involve increased thirst, frequent urination, blurred vision, and fatigue.",
    "Allergy": "Often presents with sneezing, runny nose, itching, and skin rash.",
    "Dengue": "Can show high fever, headache, joint pain, body pain, and rash.",
    "Malaria": "Frequently includes fever, chills, headache, sweating, and body pain.",
    "Typhoid": "Often linked with prolonged fever, abdominal pain, weakness, and headache.",
    "Urinary Tract Infection": "Common signs are burning urination, frequent urination, fever, and back pain.",
    "Hepatitis": "Can include fatigue, nausea, abdominal pain, and yellowing of skin.",
    "Sinusitis": "Often brings sinus pressure, headache, runny nose, and sore throat.",
    "Chickenpox": "Can present with fever, itching, body pain, and a skin rash.",
    "Measles": "Often involves fever, cough, runny nose, and widespread skin rash.",
    "Arthritis": "Commonly linked with joint pain, body pain, fatigue, and stiffness-like discomfort.",
    "Hypertension": "May be associated with headache, dizziness, chest discomfort, and fatigue.",
    "GERD": "Often includes acid reflux, chest discomfort, nausea, and bloating.",
    "Tuberculosis": "Can include cough, fever, weight loss, night sweats, and chest pain.",
    "Anemia": "Often presents with fatigue, dizziness, pale skin, shortness of breath, and headache.",
    "Heart Disease": "Often includes chest pain, shortness of breath, palpitations, fatigue, and sweating.",
    "Stroke": "Can involve confusion, dizziness, headache, blurred vision, and sudden weakness-like symptoms.",
    "Chronic Kidney Disease": "May include fatigue, swelling in legs, nausea, itching, and changes in urination.",
    "Appendicitis": "Often causes abdominal pain, nausea, vomiting, fever, and appetite loss.",
    "HIV/AIDS": "Can involve weight loss, fever, fatigue, night sweats, diarrhea, and swollen glands.",
    "Meningitis": "May present with high fever, headache, neck stiffness, vomiting, and sensitivity to light.",
    "Otitis Media": "Often includes ear pain, fever, hearing difficulty, sore throat, and headache.",
    "Psoriasis": "Can cause skin rash, itching, dry skin, and sometimes joint pain.",
    "Hypothyroidism": "Often linked with fatigue, dry skin, constipation, mood changes, and pale skin.",
    "Hyperthyroidism": "May include weight loss, palpitations, sweating, anxiety, and fatigue.",
    "COPD": "Common signs include cough, wheezing, shortness of breath, chest discomfort, and fatigue.",
}

CONDITION_THEME = {
    "Flu": ("Acute Viral Pattern", "#dbeafe", "#1d4ed8"),
    "Common Cold": ("Upper Airway Pattern", "#e0f2fe", "#0369a1"),
    "COVID-19": ("Respiratory Alert", "#fee2e2", "#b91c1c"),
    "Pneumonia": ("Pulmonary Concern", "#ede9fe", "#6d28d9"),
    "Asthma": ("Airway Reactivity", "#dcfce7", "#15803d"),
    "Bronchitis": ("Bronchial Inflammation", "#fef3c7", "#b45309"),
    "Food Poisoning": ("Digestive Distress", "#ffedd5", "#c2410c"),
    "Gastritis": ("Stomach Irritation", "#fae8ff", "#a21caf"),
    "Migraine": ("Neurological Pattern", "#ede9fe", "#7c3aed"),
    "Diabetes": ("Metabolic Pattern", "#cffafe", "#0f766e"),
    "Allergy": ("Immune Response", "#fce7f3", "#be185d"),
    "Heart Disease": ("Cardiac Risk", "#fee2e2", "#b91c1c"),
    "Stroke": ("Neurological Emergency Pattern", "#ede9fe", "#6d28d9"),
    "Chronic Kidney Disease": ("Renal Pattern", "#e0f2fe", "#0369a1"),
    "Appendicitis": ("Acute Abdominal Pattern", "#ffedd5", "#c2410c"),
    "HIV/AIDS": ("Systemic Infection Pattern", "#fdf2f8", "#be185d"),
    "Meningitis": ("CNS Alert Pattern", "#fef3c7", "#b45309"),
    "COPD": ("Chronic Respiratory Pattern", "#dcfce7", "#15803d"),
}
def format_label(value: str) -> str:
    return " ".join(str(value).replace("_", " ").split()).title()


@st.cache_data(show_spinner=False)
def load_source_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_data = pd.read_csv(DATASET_SOURCE["train_url"]).dropna(axis=1, how="all")
    test_data = pd.read_csv(DATASET_SOURCE["test_url"]).dropna(axis=1, how="all")

    train_data.columns = [str(column).strip() for column in train_data.columns]
    test_data.columns = [str(column).strip() for column in test_data.columns]

    feature_columns = [column for column in train_data.columns if column != "prognosis"]

    train_data = train_data.rename(columns={"prognosis": "disease"})
    test_data = test_data.rename(columns={"prognosis": "disease"})

    train_data = train_data.dropna(subset=["disease"]).reset_index(drop=True)
    test_data = test_data.dropna(subset=["disease"]).reset_index(drop=True)

    train_data[feature_columns] = train_data[feature_columns].fillna(0).astype(int)
    test_data = test_data.reindex(columns=["disease", *feature_columns], fill_value=0)
    test_data[feature_columns] = test_data[feature_columns].fillna(0).astype(int)

    return train_data, test_data, feature_columns


@st.cache_resource
def train_model():
    train_data, test_data, feature_columns = load_source_data()
    x_train = train_data[feature_columns]
    y_train = train_data["disease"].astype(str).str.strip()
    x_test = test_data[feature_columns]
    y_test = test_data["disease"].astype(str).str.strip()

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)

    known_label_mask = y_test.isin(encoder.classes_)
    x_test = x_test.loc[known_label_mask].reset_index(drop=True)
    y_test = y_test.loc[known_label_mask].reset_index(drop=True)
    y_test_encoded = encoder.transform(y_test)

    model = Pipeline(
        steps=[
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=16,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                ),
            )
        ]
    )
    model.fit(x_train, y_train_encoded)

    predictions = model.predict(x_test)
    decoded_true = encoder.inverse_transform(y_test_encoded)
    decoded_pred = encoder.inverse_transform(predictions)

    metrics = {
        "accuracy": accuracy_score(y_test_encoded, predictions),
        "confusion_matrix": pd.DataFrame(
            confusion_matrix(decoded_true, decoded_pred, labels=encoder.classes_),
            index=encoder.classes_,
            columns=encoder.classes_,
        ),
        "report": classification_report(
            decoded_true, decoded_pred, output_dict=True, zero_division=0
        ),
        "evaluation_cases": len(x_test),
    }
    return model, encoder, train_data, metrics, feature_columns


model, encoder, dataset, metrics, SYMPTOMS = train_model()
SYMPTOM_LABELS = {symptom: format_label(symptom) for symptom in SYMPTOMS}


def render_score_bar(label: str, value: float, color: str) -> None:
    st.markdown(
        f"""
        <div style="margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="font-weight:700; color:#ffffff;">{label}</span>
                <span style="color:#d4d4d8;">{value:.2f}%</span>
            </div>
            <div style="height:12px; background:#2a2a32; border-radius:999px; overflow:hidden;">
                <div style="width:{min(value, 100):.2f}%; height:12px; background:{color}; border-radius:999px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 14% 10%, rgba(239, 68, 68, 0.12), transparent 22%),
            radial-gradient(circle at 86% 16%, rgba(185, 28, 28, 0.12), transparent 20%),
            linear-gradient(180deg, #0b0b0d 0%, #17171b 100%);
        color: #f5f5f5;
    }
    [data-testid="stSidebar"] {display:none;}
    [data-testid="collapsedControl"] {display:none;}
    .hero {
        background:
            radial-gradient(circle at 22% 22%, rgba(255,255,255,0.10), transparent 20%),
            radial-gradient(circle at 80% 18%, rgba(255,255,255,0.08), transparent 18%),
            linear-gradient(135deg, #09090b 0%, #4a0d0d 52%, #b91c1c 100%);
        border-radius: 34px;
        padding: 38px;
        color: white;
        box-shadow: 0 28px 70px rgba(127, 29, 29, 0.28);
        margin-bottom: 24px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero h1 {
        margin: 0 0 8px 0;
        font-size: 2.7rem;
        font-weight: 800;
    }
    .hero p {
        margin: 0;
        opacity: 0.95;
        font-size: 1.03rem;
    }
    .hero-grid {
        display: grid;
        grid-template-columns: 1.7fr 1fr;
        gap: 18px;
        align-items: center;
    }
    .info-card {
        background: rgba(22, 22, 26, 0.92);
        border: 1px solid rgba(239, 68, 68, 0.12);
        border-radius: 28px;
        padding: 22px 24px;
        box-shadow: 0 16px 36px rgba(0, 0, 0, 0.24);
        margin-bottom: 18px;
    }
    .result-card {
        background:
            radial-gradient(circle at top right, rgba(239, 68, 68, 0.18), transparent 28%),
            radial-gradient(circle at bottom left, rgba(120, 13, 13, 0.16), transparent 24%),
            linear-gradient(135deg, #111114 0%, #1b1b21 100%);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-radius: 30px;
        padding: 26px;
        box-shadow: 0 20px 40px rgba(127, 29, 29, 0.24);
        margin: 14px 0 18px 0;
    }
    .result-label {
        color: #fca5a5;
        font-size: 0.9rem;
        margin-bottom: 6px;
    }
    .result-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }
    .mini-stat {
        background: rgba(20,20,24,0.94);
        border-radius: 22px;
        padding: 16px 18px;
        border: 1px solid rgba(239, 68, 68, 0.12);
        text-align: center;
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.24);
    }
    .mini-stat h3 {
        margin: 0;
        color: #ffffff;
        font-size: 1.6rem;
    }
    .mini-stat p {
        margin: 4px 0 0 0;
        color: #d4d4d8;
        font-size: 0.95rem;
    }
    .feature-pill {
        display: inline-block;
        padding: 9px 13px;
        border-radius: 999px;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.2);
        margin: 0 8px 8px 0;
        font-size: 0.9rem;
    }
    .disease-cloud {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .disease-chip {
        background: #221012;
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.14);
        border-radius: 999px;
        padding: 8px 13px;
        font-size: 0.9rem;
    }
    .selected-chip-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 14px;
    }
    .selected-chip {
        background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%);
        color: white;
        border-radius: 999px;
        padding: 9px 14px;
        font-size: 0.92rem;
        box-shadow: 0 10px 22px rgba(220, 38, 38, 0.24);
    }
    .glass-card {
        background: rgba(20,20,24,0.78);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(239, 68, 68, 0.12);
        border-radius: 28px;
        padding: 20px 22px;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.24);
        margin-bottom: 18px;
    }
    .spotlight {
        border-radius: 30px;
        padding: 26px;
        background: linear-gradient(135deg, #1a1010 0%, #2b1515 100%);
        border: 1px solid rgba(239, 68, 68, 0.16);
        box-shadow: 0 18px 38px rgba(127, 29, 29, 0.2);
    }
    .page-title {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .page-copy {
        color: #d4d4d8;
        margin-bottom: 18px;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin-top: 14px;
    }
    .feature-box {
        background: rgba(20,20,24,0.92);
        border: 1px solid rgba(239, 68, 68, 0.12);
        border-radius: 24px;
        padding: 20px;
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.24);
    }
    .feature-box h4 {
        margin: 0 0 8px 0;
        color: #ffffff;
    }
    .feature-box p {
        margin: 0;
        color: #d4d4d8;
        font-size: 0.95rem;
    }
    .insight-banner {
        background: linear-gradient(135deg, #09090b 0%, #7f1d1d 58%, #dc2626 100%);
        border-radius: 32px;
        padding: 28px;
        color: white;
        box-shadow: 0 22px 48px rgba(127, 29, 29, 0.24);
        margin-bottom: 18px;
    }
    .insight-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 16px;
        margin-top: 16px;
    }
    .metric-panel {
        background: rgba(20,20,24,0.92);
        border-radius: 24px;
        padding: 20px;
        border: 1px solid rgba(239, 68, 68, 0.12);
        box-shadow: 0 12px 26px rgba(0, 0, 0, 0.24);
    }
    .metric-panel h4 {
        margin: 0 0 8px 0;
        color: #ffffff;
    }
    .metric-panel p {
        margin: 0;
        color: #d4d4d8;
    }
    .timeline {
        border-left: 3px solid rgba(239, 68, 68, 0.45);
        padding-left: 16px;
        margin-top: 8px;
    }
    .timeline-item {
        margin-bottom: 16px;
    }
    .timeline-item h5 {
        margin: 0 0 6px 0;
        color: #ffffff;
        font-size: 1rem;
    }
    .timeline-item p {
        margin: 0;
        color: #d4d4d8;
        font-size: 0.94rem;
    }
    .page-hero {
        display: grid;
        grid-template-columns: 1.2fr 0.8fr;
        gap: 18px;
        margin-bottom: 18px;
    }
    .page-panel {
        background: rgba(20,20,24,0.9);
        border: 1px solid rgba(239, 68, 68, 0.12);
        border-radius: 28px;
        padding: 22px 24px;
        box-shadow: 0 14px 32px rgba(0, 0, 0, 0.24);
    }
    .prediction-grid {
        display: grid;
        grid-template-columns: 1.2fr 0.8fr;
        gap: 18px;
        align-items: start;
    }
    .score-card {
        background: rgba(20,20,24,0.92);
        border: 1px solid rgba(239, 68, 68, 0.12);
        border-radius: 22px;
        padding: 18px 20px;
        box-shadow: 0 12px 26px rgba(0, 0, 0, 0.24);
        margin-bottom: 14px;
    }
    .score-big {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        margin: 4px 0;
    }
    .section-note {
        color: #d4d4d8;
        font-size: 0.96rem;
        line-height: 1.6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: transparent;
        border-bottom: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: #18181b;
        border: 1px solid rgba(239, 68, 68, 0.16);
        border-radius: 16px;
        color: #fca5a5;
        padding: 10px 18px;
        box-shadow: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%) !important;
        color: #ffffff !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    .stTabs [data-testid="stTabs"] div[role="tablist"] {
        border-bottom: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-grid">
            <div>
                <h1>MediScope Symptom Checker</h1>
                <p>Review symptom patterns, surface the most likely condition, and move through a healthcare website designed to feel structured, informative, and visually polished.</p>
                <div style="margin-top:18px;">
                    <span class="feature-pill">{len(encoder.classes_)} diseases</span>
                    <span class="feature-pill">{len(SYMPTOMS)} symptoms</span>
                    <span class="feature-pill">{len(dataset)} source cases</span>
                    <span class="feature-pill">Prediction-focused interface</span>
                </div>
            </div>
            <div style="background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.14); border-radius:24px; padding:20px 22px;">
                <div style="font-size:0.86rem; opacity:0.9; text-transform:uppercase; letter-spacing:0.06em;">System profile</div>
                <div style="font-size:1.6rem; font-weight:800; margin-top:6px;">Condition intelligence view</div>
                <div style="margin-top:10px; opacity:0.92; line-height:1.65;">Built around broader symptom coverage, clearer ranking output, and focused pages for home, prediction, insights, and platform information.</div>
                <div style="margin-top:16px; display:grid; gap:10px;">
                    <div style="padding:12px 14px; border-radius:18px; background:rgba(0,0,0,0.22); border:1px solid rgba(255,255,255,0.1);">Respiratory, cardiac, digestive, infectious, endocrine, and neurological conditions</div>
                    <div style="padding:12px 14px; border-radius:18px; background:rgba(0,0,0,0.22); border:1px solid rgba(255,255,255,0.1);">Separate pages for prediction guidance, clinical insights, and platform information</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.empty()

tab_home, tab_predict, tab_insights, tab_about = st.tabs(["Home", "Prediction", "Insights", "About"])

with tab_home:
    st.markdown('<div class="page-title">Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-copy">A structured symptom-based healthcare website with broader disease coverage, richer symptom input, and dedicated pages for prediction, insights, and platform information.</div>',
        unsafe_allow_html=True,
    )

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    with summary_col1:
        st.markdown(
            f'<div class="mini-stat"><h3>{len(dataset)}</h3><p>Total Cases</p></div>',
            unsafe_allow_html=True,
        )
    with summary_col2:
        st.markdown(
            f'<div class="mini-stat"><h3>{len(encoder.classes_)}</h3><p>Conditions Covered</p></div>',
            unsafe_allow_html=True,
        )
    with summary_col3:
        st.markdown(
            f'<div class="mini-stat"><h3>{len(SYMPTOMS)}</h3><p>Symptoms Available</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-box">
                <h4>Condition Coverage</h4>
                <p>The platform spans respiratory, cardiovascular, infectious, digestive, endocrine, dermatological, renal, and neurological disease groups to create a broader prediction space.</p>
            </div>
            <div class="feature-box">
                <h4>Source Dataset</h4>
                <p>The prediction engine now trains on a real disease-symptom dataset loaded from public source files instead of internally generated synthetic cases.</p>
            </div>
            <div class="feature-box">
                <h4>Website Structure</h4>
                <p>The home page now leads into dedicated website pages for clinical prediction, model insights, and platform information with a cleaner flow.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Dataset source: {DATASET_SOURCE['name']} | Origin: {DATASET_SOURCE['origin_url']} | Train CSV: {DATASET_SOURCE['train_url']}"
    )

with tab_predict:
    st.markdown(
        """
        <div class="page-hero">
            <div class="page-panel">
                <div class="page-title" style="margin-bottom:0;">Prediction Studio</div>
                <div class="page-copy" style="margin-top:8px; margin-bottom:0;">Select symptoms directly on the page, then review the main match and the ranked score profile.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_symptoms = st.multiselect(
        "Select observed symptoms",
        options=SYMPTOMS,
        format_func=lambda symptom: SYMPTOM_LABELS[symptom],
        placeholder="Choose symptoms here...",
    )
    predict_clicked = st.button("Predict Disease", use_container_width=True)

    left, right = st.columns([1.12, 0.88])
    with left:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="margin-top:0; color:#ffffff;">Selected Symptoms</h3>
                <p style="color:#d4d4d8; margin-bottom:0;">Build the patient profile from this section and review the ranked disease response below.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if selected_symptoms:
            selected_symptom_html = "".join(
                [f'<span class="selected-chip">{SYMPTOM_LABELS[symptom]}</span>' for symptom in selected_symptoms]
            )
            st.markdown(f'<div class="selected-chip-wrap">{selected_symptom_html}</div>', unsafe_allow_html=True)
        else:
            st.info("Choose one or more symptoms from the list above.")

        if predict_clicked:
            input_df = pd.DataFrame([{symptom: int(symptom in selected_symptoms) for symptom in SYMPTOMS}])
            prediction_encoded = model.predict(input_df)[0]
            prediction = encoder.inverse_transform([prediction_encoded])[0]
            probabilities = model.predict_proba(input_df)[0]
            top_indices = probabilities.argsort()[::-1][:5]
            top_probability = float(probabilities[top_indices[0]]) * 100
            theme_label, theme_bg, theme_fg = CONDITION_THEME.get(
                prediction, ("Predicted Profile", "#fce7f3", "#9d174d")
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Most likely disease</div>
                    <p class="result-value">{prediction}</p>
                    <div class="result-label">Confidence score: {top_probability:.2f}%</div>
                    <div class="result-label">{DISEASE_INFO.get(prediction, "Predicted from the selected symptom pattern.")}</div>
                    <div style="margin-top:14px;">
                        <span style="background:{theme_bg}; color:{theme_fg}; padding:9px 14px; border-radius:999px; font-size:0.9rem; font-weight:700;">{theme_label}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Prediction Breakdown")
            score_left, score_right = st.columns([1.05, 0.95])
            with score_left:
                st.markdown(
                    f"""
                    <div class="score-card">
                        <div class="result-label">Primary Match</div>
                        <div class="score-big">{prediction}</div>
                        <div class="section-note">{DISEASE_INFO.get(prediction, "Predicted from the selected symptom pattern.")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="score-card">
                        <div class="result-label">Confidence</div>
                        <div class="score-big">{top_probability:.2f}%</div>
                        <div class="section-note">This score reflects how strongly the selected symptom pattern matches the leading disease class in the current model.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with score_right:
                st.markdown("##### Top Condition Scores")
                score_colors = ["#dc2626", "#ef4444", "#f97316", "#7c2d12", "#991b1b"]
                for idx, disease_index in enumerate(top_indices):
                    disease_name = encoder.inverse_transform([disease_index])[0]
                    render_score_bar(
                        disease_name,
                        float(probabilities[disease_index]) * 100,
                        score_colors[idx % len(score_colors)],
                    )

            probability_table = pd.DataFrame(
                {
                    "Disease": encoder.inverse_transform(top_indices),
                    "Probability (%)": [round(float(probabilities[i]) * 100, 2) for i in top_indices],
                }
            )
            st.markdown("#### Ranked Conditions")
            st.dataframe(probability_table, width="stretch", hide_index=True)
            st.caption("Use this output as a clinical-style prediction summary, not as a medical diagnosis.")
        else:
            st.markdown(
                """
                <div class="spotlight">
                    <div style="font-size:0.9rem; color:#fca5a5; font-weight:700;">Prediction Canvas</div>
                    <div style="font-size:1.5rem; color:#ffffff; font-weight:800; margin-top:6px;">Ready for symptom analysis</div>
                    <div style="color:#d4d4d8; margin-top:8px;">Once you choose symptoms and press Predict Disease, this area will transform into the diagnostic result card with ranked conditions.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with right:
        st.markdown(
            """
            <div class="info-card">
                <h3 style="margin-top:0; color:#ffffff;">Case Builder Notes</h3>
                <p style="color:#d4d4d8; margin-bottom:10px;">Choose only the symptoms that are actually observed. The model now uses the symptom columns from the source dataset, so fewer but more relevant selections usually give clearer rankings.</p>
                <div class="disease-cloud">
                    <span class="disease-chip">Source-backed symptoms</span>
                    <span class="disease-chip">Binary symptom columns</span>
                    <span class="disease-chip">Observed signs only</span>
                    <span class="disease-chip">Cleaner ranking</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="info-card">
                <h3 style="margin-top:0; color:#12355b;">Prediction Quality</h3>
                <p class="section-note">For clearer results, combine core symptoms with one or two support symptoms rather than selecting every symptom at once. This usually creates a cleaner disease ranking and a stronger main result card.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_insights:
    st.markdown(
        f"""
            <div class="page-hero">
                <div class="insight-banner">
                    <div style="font-size:0.92rem; opacity:0.92;">Clinical Insights</div>
                    <div style="font-size:2rem; font-weight:800; margin-top:6px;">How this prediction system is structured</div>
                    <div style="margin-top:10px; max-width:760px; opacity:0.96;">This page explains the model, symptom coverage, and how the platform interprets symptom combinations across a broad condition library.</div>
                </div>
                <div class="page-panel">
                    <div style="font-size:0.86rem; color:#fca5a5; font-weight:700;">Snapshot</div>
                    <div style="font-size:1.2rem; color:#ffffff; font-weight:800; margin-top:6px;">{metrics['accuracy'] * 100:.2f}% measured accuracy</div>
                    <div style="color:#d4d4d8; margin-top:8px;">Measured on the source dataset test split loaded from the public CSV files.</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="insight-grid">
            <div class="metric-panel">
                <h4>Prediction Engine</h4>
                <p>Algorithm: Random Forest Classifier</p>
                <p>Total training cases: {len(dataset)}</p>
                <p>Evaluation cases: {metrics['evaluation_cases']}</p>
                <p>Diseases covered: {len(encoder.classes_)}</p>
                <p>Symptoms modeled: {len(SYMPTOMS)}</p>
            </div>
            <div class="metric-panel">
                <h4>Coverage Summary</h4>
                <p>Includes the disease labels and symptom columns provided by the source training data.</p>
                <p>Feature coverage now follows the public dataset schema rather than a hand-built synthetic profile list.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    insights_left, insights_right = st.columns(2)
    with insights_left:
        st.markdown(
            """
            <div class="info-card">
                <h3 style="margin-top:0; color:#ffffff;">How The Model Interprets Symptoms</h3>
                <p class="section-note">Each selected symptom becomes a binary feature in the input vector. The classifier compares that pattern against the real disease cases loaded from the source CSV files and estimates the nearest disease matches based on the trained forest.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with insights_right:
        st.markdown(
            """
            <div class="info-card">
                <h3 style="margin-top:0; color:#ffffff;">Why More Specific Symptoms Matter</h3>
                <p class="section-note">Generic symptoms occur across many diseases, so they do not separate classes very well on their own. Adding the most relevant observed symptoms from the dataset columns usually narrows the prediction more effectively.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        <div class="info-card">
            <h3 style="margin-top:0; color:#ffffff;">Interpretation Notes</h3>
            <p class="section-note">The score shown in the prediction page is a model confidence estimate within the current environment, not a clinical certainty score. The platform is designed to present classification workflow, symptom reasoning, and healthcare interface quality in a clear and accessible format.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_about:
    st.markdown(
        """
            <div class="page-hero">
                <div class="insight-banner">
                    <div style="font-size:0.92rem; opacity:0.92;">About MediScope</div>
                    <div style="font-size:2rem; font-weight:800; margin-top:6px;">A classification-based disease prediction platform</div>
                    <div style="margin-top:10px; max-width:760px; opacity:0.96;">MediScope combines machine learning with a modern healthcare-style interface to create a disease prediction experience that is structured, informative, and visually refined.</div>
                </div>
                <div class="page-panel">
                    <div style="font-size:0.86rem; color:#fca5a5; font-weight:700;">Platform Positioning</div>
                    <div style="font-size:1.2rem; color:#ffffff; font-weight:800; margin-top:6px;">Built as a polished healthcare web experience</div>
                    <div style="color:#d4d4d8; margin-top:8px;">The focus is not only prediction, but also a stronger website experience with cleaner information flow and richer symptom coverage.</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-box">
                <h4>Clinical Focus</h4>
                <p>Support symptom-based condition screening through a polished interface that feels like a real healthcare website.</p>
            </div>
            <div class="feature-box">
                <h4>Prediction Engine</h4>
                <p>The app uses classification to map patient symptom patterns to the most likely disease classes and rank the top matches.</p>
            </div>
            <div class="feature-box">
                <h4>Website Experience</h4>
                <p>The interface uses a cleaner webpage flow with in-page section selection rather than sidebar navigation.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    about_left, about_right = st.columns(2)
    with about_left:
        st.markdown(
            """
            <div class="info-card">
                <h3 style="margin-top:0; color:#ffffff;">Why MediScope</h3>
                <p class="section-note">MediScope brings together a disease classification engine and a healthcare-style presentation layer in one platform. It is designed to make symptom-driven screening feel clearer, more structured, and easier to interpret.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with about_right:
        st.markdown(
            """
            <div class="info-card">
                <h3 style="margin-top:0; color:#ffffff;">Service Scope</h3>
                <p class="section-note">The current experience focuses on symptom-based disease ranking across a wide set of globally relevant conditions. It emphasizes readability, ranking clarity, and page-level navigation rather than patient record management.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        f"""
        <div class="info-card">
            <h3 style="margin-top:0; color:#ffffff;">Dataset Source</h3>
            <p class="section-note">Current source dataset: {DATASET_SOURCE['name']}</p>
            <p class="section-note">Origin: {DATASET_SOURCE['origin_url']}</p>
            <p class="section-note">Training CSV: {DATASET_SOURCE['train_url']}</p>
            <p class="section-note">Test CSV: {DATASET_SOURCE['test_url']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-card">
            <h3 style="margin-top:0; color:#ffffff;">Service Roadmap</h3>
            <div class="timeline">
                <div class="timeline-item">
                    <h5>1. Expanded Clinical Data</h5>
                    <p>Expand beyond the current source dataset with additional curated clinical datasets and cleaner label harmonization.</p>
                </div>
                <div class="timeline-item">
                    <h5>2. Report Generation</h5>
                    <p>Add downloadable patient summary cards or PDF-style reports for stronger reporting and clinical-style presentation.</p>
                </div>
                <div class="timeline-item">
                    <h5>3. Role-Based Experience</h5>
                    <p>Introduce patient, admin, or doctor-facing views to expand the platform into a more complete digital health experience.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
