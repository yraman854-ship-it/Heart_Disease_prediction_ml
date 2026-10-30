# import json
# from pathlib import Path
# import joblib
# import numpy as np
# import pandas as pd
# import streamlit as st

# st.set_page_config(page_title="Heart Disease AI", page_icon="❤️", layout="wide")

# BASE = Path(__file__).resolve().parent
# MODEL = joblib.load(BASE/"models"/"best_model.joblib")
# SUMMARY = json.load(open(BASE/"results"/"summary.json",encoding="utf-8"))
# COMPARISON = pd.read_csv(BASE/"results"/"model_comparison.csv")
# IMPORTANCE = pd.read_csv(BASE/"results"/"feature_importance.csv")

# st.sidebar.title("❤️ Heart Disease AI")
# page = st.sidebar.radio("Navigation",["Patient Prediction","Dashboard","Model Performance","About"])
# st.sidebar.markdown("---")
# st.sidebar.caption("Academic demonstration • Random Forest")

# if page == "Patient Prediction":
#     st.title("❤️ Heart Disease Risk Prediction")
#     st.write("Enter patient information to demonstrate the trained classification model.")

#     with st.form("patient_form"):
#         a,b,c = st.columns(3)

#         with a:
#             country = st.selectbox("Country", SUMMARY["countries"])
#             age_years = st.number_input("Age (years)",18,100,50)
#             gender = st.selectbox("Gender code",[1,2])
#             active = st.selectbox("Physical activity (0/1)",[0,1],index=1)
#             occupation = st.selectbox("Occupation",SUMMARY["occupations"])

#         with b:
#             ap_hi = st.number_input("Systolic BP (ap_hi)",70,250,130)
#             ap_lo = st.number_input("Diastolic BP (ap_lo)",40,200,80)
#             cholesterol = st.selectbox("Cholesterol code",[1,2,3])
#             gluc = st.selectbox("Glucose code",[1,2,3])
#             smoke = st.selectbox("Smoking (0/1)",[0,1])

#         with c:
#             alco = st.selectbox("Alcohol intake (0/1)",[0,1])
#             height = st.number_input("Height (cm)",120,220,170)
#             weight = st.number_input("Weight (kg)",30.0,200.0,70.0,step=0.5)
#             bmi = weight/((height/100)**2)
#             pulse_pressure = ap_hi-ap_lo
#             st.metric("BMI",f"{bmi:.1f}")
#             st.metric("Pulse Pressure",f"{pulse_pressure:.0f}")

#         submitted = st.form_submit_button("🔍 Predict",use_container_width=True)

#     if submitted:
#         patient = pd.DataFrame([{
#             "country":country,"active":active,"age_years":age_years,"alco":alco,
#             "ap_hi":ap_hi,"ap_lo":ap_lo,"cholesterol":cholesterol,"gender":gender,
#             "gluc":gluc,"height":height,"occupation":occupation,"smoke":smoke,
#             "weight":weight,"bmi":bmi,"pulse_pressure":pulse_pressure
#         }])
#         pred = int(MODEL.predict(patient)[0])
#         prob = float(MODEL.predict_proba(patient)[0,1])

#         st.markdown("---")
#         if pred == 1:
#             st.error(f"### Higher predicted risk\nModel probability for disease = 1: **{prob:.1%}**")
#         else:
#             st.success(f"### Lower predicted risk\nModel probability for disease = 1: **{prob:.1%}**")
#         st.progress(min(max(prob,0),1))
#         st.caption("This is a dataset-specific model estimate for academic demonstration, not a diagnosis.")

# elif page == "Dashboard":
#     st.title("📊 Project Dashboard")
#     a,b,c,d = st.columns(4)
#     a.metric("Patients",f"{SUMMARY['rows']:,}")
#     b.metric("Features",SUMMARY["columns"])
#     c.metric("Disease = 1",f"{SUMMARY['disease_1']:,}")
#     d.metric("Disease = 0",f"{SUMMARY['disease_0']:,}")

#     st.subheader("Final model metrics")
#     metric_df = pd.DataFrame({
#         "Metric":["Accuracy","Precision","Recall","F1-Score","ROC-AUC"],
#         "Score":[SUMMARY["metrics"]["accuracy"],SUMMARY["metrics"]["precision"],
#                  SUMMARY["metrics"]["recall"],SUMMARY["metrics"]["f1"],SUMMARY["metrics"]["roc_auc"]]
#     }).set_index("Metric")
#     st.bar_chart(metric_df)

#     st.subheader("Top predictive features")
#     top = IMPORTANCE.head(10).copy()
#     top["Feature"] = top["Feature"].str.replace("num__","",regex=False).str.replace("cat__","",regex=False)
#     st.bar_chart(top.set_index("Feature")["Importance"])

# elif page == "Model Performance":
#     st.title("📈 Model Performance")
#     st.dataframe(COMPARISON.style.format({
#         "Accuracy":"{:.3f}","Precision":"{:.3f}","Recall":"{:.3f}",
#         "F1-Score":"{:.3f}","ROC-AUC":"{:.3f}"
#     }),use_container_width=True,hide_index=True)
#     st.bar_chart(COMPARISON.set_index("Model")[["Accuracy","Precision","Recall","F1-Score"]])
#     st.info("Recall is especially important in health-risk screening because false negatives can be consequential.")

# else:
#     st.title("ℹ️ About the Project")
#     st.markdown("""
#     ### Objective
#     Predict the binary `disease` target using patient, lifestyle and health measurements.

#     ### Workflow
#     1. Data exploration
#     2. Data quality checks
#     3. Feature engineering: age in years, BMI and pulse pressure
#     4. Missing-value handling and scaling
#     5. Categorical encoding
#     6. Train/test split
#     7. Logistic Regression, Decision Tree, Random Forest and Linear SVM
#     8. Evaluation with Accuracy, Precision, Recall, F1-score and ROC-AUC
#     9. Feature-importance analysis
#     10. Interactive Streamlit presentation

#     ### Important
#     This is an academic demonstration. It is not a medical device and should not be used for diagnosis, treatment or emergency decisions.
#     """)

# COMPLETE UPDATED STREAMLIT UI
# Heart Disease Detection Project
# ------------------------------------------------------------
# Expected project structure:
# Heart_Disease_Detection/
# ├── app.py
# ├── Data_file.xlsx
# ├── requirements.txt
# └── models/
#     └── best_model.joblib
#
# The app keeps the trained ML model unchanged and provides
# a presentation-ready Streamlit interface around it.

# COMPLETE UPDATED STREAMLIT UI
# Heart Disease Detection Project
# ------------------------------------------------------------
# Expected project structure:
# Heart_Disease_Detection/
# ├── app.py
# ├── Data_file.xlsx
# ├── requirements.txt
# └── models/
#     └── best_model.joblib
#
# The app keeps the trained ML model unchanged and provides
# a presentation-ready Streamlit interface around it.

import os

from pathlib import Path

from datetime import datetime



import joblib

import numpy as np

import pandas as pd

import streamlit as st

import matplotlib.pyplot as plt


def render_table(data, title=None, precision=None):
    """Render pandas data as HTML without Streamlit's PyArrow-dependent dataframe widget."""
    if data is None:
        return
    table = data.copy() if hasattr(data, "copy") else data
    if hasattr(table, "to_frame") and not hasattr(table, "columns"):
        table = table.to_frame()
    if precision is not None and hasattr(table, "round"):
        table = table.round(precision)
    if title:
        st.markdown(f"**{title}**")
    st.markdown(table.to_html(index=True, border=0, classes="dataframe", escape=True), unsafe_allow_html=True)




# ============================================================

# PAGE CONFIG

# ============================================================

st.set_page_config(

    page_title="Heart Disease Detection",

    page_icon="❤️",

    layout="wide",

    initial_sidebar_state="expanded",

)





# ============================================================

# PATHS

# ============================================================

BASE_DIR = Path(__file__).resolve().parent



DATA_CANDIDATES = [

    BASE_DIR / "Data_file.xlsx",

    BASE_DIR / "data" / "Data_file.xlsx",

    BASE_DIR / "Data_file.csv",

]



MODEL_CANDIDATES = [

    BASE_DIR / "models" / "best_model.joblib",

    BASE_DIR / "best_model.joblib",

    BASE_DIR / "models" / "heart_disease_model.joblib",

]





# ============================================================

# CUSTOM CSS

# ============================================================

st.markdown(

    """

    <style>



    .main-title {

        padding: 1.4rem 1.6rem;

        border-radius: 18px;

        background: linear-gradient(135deg, #b91c1c, #ef4444);

        color: white;

        margin-bottom: 1.2rem;

    }



    .main-title h1 {

        margin: 0;

        font-size: 2.35rem;

    }



    .main-title p {

        margin: .35rem 0 0;

        font-size: 1.02rem;

        opacity: .95;

    }



    .section-card {

        padding: 1rem 1.2rem;

        border: 1px solid rgba(128,128,128,.22);

        border-radius: 15px;

        margin-bottom: 1rem;

    }



    .result-card {

        padding: 1.5rem;

        border-radius: 18px;

        text-align: center;

        border: 1px solid rgba(128,128,128,.25);

        margin: .8rem 0 1rem;

    }



    .result-big {

        font-size: 2rem;

        font-weight: 800;

        margin: .3rem 0;

    }



    .muted {

        opacity: .72;

        font-size: .9rem;

    }



    .footer {

        text-align: center;

        opacity: .65;

        padding: 1.2rem 0 .4rem;

        font-size: .85rem;

    }



    </style>

    """,

    unsafe_allow_html=True,

)





# ============================================================

# HELPER FUNCTIONS

# ============================================================

def find_existing(paths):

    """Return the first existing path."""

    for path in paths:

        if path.exists():

            return path

    return None





@st.cache_data

def load_data(path_str):

    """Load Excel or CSV dataset."""

    path = Path(path_str)



    if path.suffix.lower() == ".csv":

        return pd.read_csv(path)



    return pd.read_excel(path)





@st.cache_resource

def load_model(path_str):

    """Load the already-trained model."""

    return joblib.load(path_str)





def safe_mode(series, default=None):

    """Return the most common non-null value."""

    values = series.dropna()



    if values.empty:

        return default



    return values.mode().iloc[0]





def get_model_feature_names(model):

    """

    Best-effort extraction of transformed feature names.

    Works with many sklearn pipelines and preprocessors.

    """



    try:

        if hasattr(model, "get_feature_names_out"):

            return list(model.get_feature_names_out())

    except Exception:

        pass



    try:

        if hasattr(model, "named_steps"):

            preprocessor = model.named_steps.get("preprocessor")



            if (

                preprocessor is not None

                and hasattr(preprocessor, "get_feature_names_out")

            ):

                return list(preprocessor.get_feature_names_out())



    except Exception:

        pass



    return None





def get_final_estimator(model):

    """Find the final estimator inside a sklearn Pipeline."""



    if hasattr(model, "named_steps"):



        for step in reversed(list(model.named_steps.values())):



            if (

                hasattr(step, "feature_importances_")

                or hasattr(step, "coef_")

            ):

                return step



    return model





def get_feature_importance_table(model):

    """Create a feature-importance table when the model supports it."""



    estimator = get_final_estimator(model)



    names = get_model_feature_names(model)



    if hasattr(estimator, "feature_importances_"):



        values = np.asarray(

            estimator.feature_importances_,

            dtype=float

        )



    elif hasattr(estimator, "coef_"):



        coef = np.asarray(estimator.coef_)



        values = np.abs(

            coef[0] if coef.ndim > 1 else coef

        )



    else:

        return None



    if names is None or len(names) != len(values):



        names = [

            f"Feature {i + 1}"

            for i in range(len(values))

        ]



    result = pd.DataFrame(

        {

            "Feature": names,

            "Importance": values,

        }

    )



    return (

        result

        .sort_values("Importance", ascending=False)

        .head(12)

    )





def build_input_row(

    df,

    age_years,

    gender,

    country,

    occupation,

    active,

    alco,

    ap_hi,

    ap_lo,

    cholesterol,

    gluc,

    height_cm,

    smoke,

    weight_kg,

):

    """

    Build a prediction row.



    The original dataset uses age in days, while the UI accepts

    age in years. BMI and pulse pressure are added as engineered

    features.



    Extra original columns such as date/id are retained when

    present because some saved pipelines may expect them.

    """



    age_days = float(age_years) * 365.25



    bmi = (

        float(weight_kg)

        / ((float(height_cm) / 100.0) ** 2)

    )



    pulse_pressure = (

        float(ap_hi) - float(ap_lo)

    )



    row = {

        "age": age_days,

        "age_years": float(age_years),

        "gender": gender,

        "country": country,

        "occupation": occupation,

        "active": int(active),

        "alco": int(alco),

        "ap_hi": float(ap_hi),

        "ap_lo": float(ap_lo),

        "cholesterol": int(cholesterol),

        "gluc": int(gluc),

        "height": float(height_cm),

        "weight": float(weight_kg),

        "smoke": int(smoke),

        "bmi": float(bmi),

        "pulse_pressure": float(pulse_pressure),

    }



    # Keep date if the original dataset contains it.

    if "date" in df.columns:



        mode_date = safe_mode(

            df["date"],

            default=datetime.now().strftime("%Y-%m-%d")

        )



        row["date"] = mode_date



    # Keep id if present. The training pipeline should normally

    # exclude it, but retaining it prevents missing-column errors

    # for pipelines that expect the original dataframe schema.

    if "id" in df.columns:



        max_id = pd.to_numeric(

            df["id"],

            errors="coerce"

        ).max()



        row["id"] = (

            0

            if pd.isna(max_id)

            else int(max_id) + 1

        )



    # Add any other original columns that may be expected.

    for col in df.columns:



        if col in row or col == "disease":

            continue



        if pd.api.types.is_numeric_dtype(df[col]):



            value = pd.to_numeric(

                df[col],

                errors="coerce"

            ).median()



            row[col] = (

                0

                if pd.isna(value)

                else value

            )



        else:



            row[col] = safe_mode(

                df[col],

                default="Unknown"

            )



    return pd.DataFrame([row])





def model_predict(model, input_df):

    """Run prediction and return class + class-1 probability."""



    prediction = int(

        np.asarray(

            model.predict(input_df)

        ).ravel()[0]

    )



    probability = None



    if hasattr(model, "predict_proba"):



        proba = model.predict_proba(input_df)[0]



        if len(proba) >= 2:

            probability = float(proba[1])

        else:

            probability = float(proba[0])



    return prediction, probability





# ============================================================

# LOAD DATA AND MODEL

# ============================================================

data_path = find_existing(DATA_CANDIDATES)

model_path = find_existing(MODEL_CANDIDATES)



if data_path is None:



    st.error("Data_file.xlsx was not found.")



    st.info(

        "Place Data_file.xlsx in the same folder as app.py."

    )



    st.stop()





if model_path is None:



    st.error("best_model.joblib was not found.")



    st.info(

        "Place your trained model at "

        "models/best_model.joblib."

    )



    st.stop()





try:



    df = load_data(str(data_path))



except Exception as exc:



    st.error(

        f"Could not load the dataset: {exc}"

    )



    st.stop()





try:



    model = load_model(str(model_path))



except Exception as exc:



    st.error(

        f"Could not load the trained model: {exc}"

    )



    st.stop()





# ============================================================

# HEADER

# ============================================================

# st.markdown(

#     """

#     <div class="main-title">



#         <h1>

#             ❤️ Heart Disease Detection

#         </h1>



#         <p>

#             Machine-learning based cardiovascular disease

#             classification dashboard

#         </p>



#     </div>

#     """,

#     unsafe_allow_html=True,

# )



# ============================================================

# HEADER

# ============================================================



st.title("❤️ Heart Disease Detection")



st.caption(

    "Machine-learning based cardiovascular disease classification dashboard"

)



st.markdown("---")





# ============================================================

# SIDEBAR

# ============================================================

st.sidebar.title("❤️ Heart AI")



st.sidebar.caption(

    "Interactive ML presentation dashboard"

)



st.sidebar.markdown("---")



page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Dashboard",

        "🔍 Patient Prediction",

        "📊 Model Performance",

        "📁 Data Explorer",

        "ℹ️ About Project",

    ],

)



st.sidebar.markdown("---")



st.sidebar.caption(

    "Final model: Random Forest"

)



st.sidebar.caption(

    "Academic project • Streamlit UI"

)





# ============================================================

# COMMON DATA VALUES

# ============================================================

target_exists = "disease" in df.columns



dataset_size = len(df)



if "country" in df.columns:



    country_values = sorted(

        df["country"]

        .dropna()

        .astype(str)

        .unique()

        .tolist()

    )



else:



    country_values = ["Unknown"]





if "occupation" in df.columns:



    occupation_values = sorted(

        df["occupation"]

        .dropna()

        .astype(str)

        .unique()

        .tolist()

    )



else:



    occupation_values = ["Unknown"]





if "gender" in df.columns:



    gender_values = sorted(

        df["gender"]

        .dropna()

        .astype(str)

        .unique()

        .tolist()

    )



else:



    gender_values = [

        "Female",

        "Male",

    ]





# ============================================================

# DASHBOARD

# ============================================================

if page == "🏠 Dashboard":



    st.subheader("📊 Project Dashboard")



    st.caption(

        "Quick overview of the dataset, selected model, "

        "and classification results."

    )



    c1, c2, c3, c4 = st.columns(4)



    with c1:



        st.metric(

            "📋 Dataset Records",

            f"{dataset_size:,}"

        )



    with c2:



        st.metric(

            "🧠 Final Model",

            "Random Forest"

        )



    with c3:



        st.metric(

            "🎯 Accuracy",

            "73.12%"

        )



    with c4:



        st.metric(

            "📈 ROC-AUC",

            "0.797"

        )



    st.markdown("---")



    left, right = st.columns(

        [1.1, 1]

    )



    with left:



        st.subheader(

            "🩺 Disease Distribution"

        )



        if target_exists:



            counts = (

                df["disease"]

                .value_counts()

                .sort_index()

            )



            display_counts = pd.DataFrame(

                {

                    "Class": [

                        "No Disease (0)",

                        "Disease (1)",

                    ][:len(counts)],



                    "Records": counts.values,

                }

            )



            fig, ax = plt.subplots(figsize=(7, 4))



            ax.bar(

                display_counts["Class"],

                display_counts["Records"]

            )



            ax.set_title("Disease Class Distribution")

            ax.set_xlabel("Disease Class")

            ax.set_ylabel("Number of Records")



            for i, value in enumerate(display_counts["Records"]):

                ax.text(

                    i,

                    value,

                    f"{value:,}",

                    ha="center",

                    va="bottom"

            )



            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)



            st.caption(

                "Target variable: disease "

                "(0 = no disease, 1 = disease)."

            )



        else:



            st.warning(

                "Target column 'disease' "

                "was not found."

            )





    with right:



        st.subheader("🔄 ML Workflow")



        st.markdown(

            """

            **1. Data Collection**



            Patient-level cardiovascular records



            **2. Preprocessing**



            Encoding + scaling + consistent pipeline



            **3. Feature Engineering**



            Age in years, BMI and pulse pressure



            **4. Model Training**



            Logistic Regression, Decision Tree,

            Random Forest and Linear SVM



            **5. Final Model**



            Random Forest selected from the tested models



            **6. Streamlit Interface**



            Interactive prediction and analysis

            """

        )





    st.markdown("---")



    st.subheader(

        "📌 Dataset Snapshot"

    )



    info_cols = st.columns(4)



    with info_cols[0]:



        st.metric(

            "Columns",

            df.shape[1]

        )



    with info_cols[1]:



        st.metric(

            "Missing Cells",

            f"{int(df.isna().sum().sum()):,}"

        )



    with info_cols[2]:



        st.metric(

            "Duplicate Rows",

            f"{int(df.duplicated().sum()):,}"

        )



    with info_cols[3]:



        st.metric(

            "Target Available",

            "Yes" if target_exists else "No"

        )





    st.markdown("---")



    st.subheader(

        "⭐ Why Random Forest?"

    )



    st.success(

        "Random Forest achieved the best overall result "

        "among the four tested classification models in "

        "this project, with 73.12% accuracy and 71.97% "

        "F1-score."

    )





# ============================================================

# PATIENT PREDICTION

# ============================================================

elif page == "🔍 Patient Prediction":



    st.subheader(

        "🔍 Patient Prediction"

    )



    st.caption(

        "Enter patient information and run "

        "the trained classification model."

    )



    with st.form("patient_form"):



        st.markdown(

            "### 👤 Patient Information"

        )



        a, b, c = st.columns(3)



        with a:



            age_years = st.number_input(

                "Age (years)",

                1,

                120,

                50

            )



        with b:



            gender = st.selectbox(

                "Gender",

                gender_values

            )



        with c:



            country = st.selectbox(

                "Country",

                country_values

            )





        d, e, f = st.columns(3)



        with d:



            occupation = st.selectbox(

                "Occupation",

                occupation_values

            )



        with e:



            height_cm = st.number_input(

                "Height (cm)",

                100.0,

                230.0,

                170.0,

                1.0

            )



        with f:



            weight_kg = st.number_input(

                "Weight (kg)",

                25.0,

                250.0,

                70.0,

                0.5

            )





        st.markdown(

            "### 🩺 Blood Pressure & Health"

        )



        g, h, i = st.columns(3)



        with g:



            ap_hi = st.number_input(

                "Systolic BP (ap_hi)",

                70,

                250,

                120

            )



        with h:



            ap_lo = st.number_input(

                "Diastolic BP (ap_lo)",

                40,

                160,

                80

            )



        with i:



            cholesterol_label = st.selectbox(

                "Cholesterol",

                [

                    "Normal",

                    "Above Normal",

                    "Well Above Normal",

                ]

            )





        j, k, l = st.columns(3)



        with j:



            glucose_label = st.selectbox(

                "Glucose",

                [

                    "Normal",

                    "Above Normal",

                    "Well Above Normal",

                ]

            )



        with k:



            smoke_label = st.selectbox(

                "Smoking",

                [

                    "No",

                    "Yes",

                ]

            )



        with l:



            active_label = st.selectbox(

                "Physical Activity",

                [

                    "No",

                    "Yes",

                ]

            )





        m, n, o = st.columns(3)



        with m:



            alco_label = st.selectbox(

                "Alcohol Intake",

                [

                    "No",

                    "Yes",

                ]

            )



        with n:



            bmi_value = (

                weight_kg

                / ((height_cm / 100) ** 2)

            )



            st.metric(

                "Calculated BMI",

                f"{bmi_value:.2f}"

            )



        with o:



            st.metric(

                "Pulse Pressure",

                f"{ap_hi - ap_lo:.0f} mmHg"

            )





        st.markdown("---")



        submitted = st.form_submit_button(

            "🔍 Analyze Patient",

            use_container_width=True

        )





    if submitted:



        cholesterol = {

            "Normal": 1,

            "Above Normal": 2,

            "Well Above Normal": 3,

        }[cholesterol_label]





        gluc = {

            "Normal": 1,

            "Above Normal": 2,

            "Well Above Normal": 3,

        }[glucose_label]





        smoke = (

            1

            if smoke_label == "Yes"

            else 0

        )



        active = (

            1

            if active_label == "Yes"

            else 0

        )



        alco = (

            1

            if alco_label == "Yes"

            else 0

        )





        input_df = build_input_row(

            df=df,

            age_years=age_years,

            gender=gender,

            country=country,

            occupation=occupation,

            active=active,

            alco=alco,

            ap_hi=ap_hi,

            ap_lo=ap_lo,

            cholesterol=cholesterol,

            gluc=gluc,

            height_cm=height_cm,

            smoke=smoke,

            weight_kg=weight_kg,

        )





        try:



            prediction, probability = model_predict(

                model,

                input_df

            )



        except Exception as exc:



            st.error(

                "Prediction failed because the "

                "UI inputs do not match the saved model."

            )



            st.code(str(exc))



            with st.expander(

                "Debug: data sent to model"

            ):



                render_table(input_df)



            st.stop()





        st.markdown("---")



        st.subheader(

            "❤️ Prediction Result"

        )



        r1, r2 = st.columns(

            [1.2, 1]

        )





        with r1:



            if prediction == 1:



                st.error(

                    """

                    ### 🔴 Model Prediction: Disease Class



                    The model classified this input as

                    **Class 1**.

                    """

                )



            else:



                st.success(

                    """

                    ### 🟢 Model Prediction: No Disease Class



                    The model classified this input as

                    **Class 0**.

                    """

                )





        with r2:



            if probability is not None:



                st.metric(

                    "Model Probability for Class 1",

                    f"{probability * 100:.2f}%"

                )



                st.progress(

                    min(

                        max(

                            probability,

                            0.0

                        ),

                        1.0

                    )

                )



            else:



                st.info(

                    "The saved model does not "

                    "expose probability estimates."

                )





        st.caption(

            "This probability is the model's estimated "

            "probability for class 1 based on patterns "

            "learned from the training data. It is not "

            "a medical diagnosis."

        )





        st.markdown("---")



        st.subheader(

            "📋 Patient Summary"

        )



        s1, s2, s3 = st.columns(3)





        with s1:



            st.write(

                f"**Age:** {age_years} years"

            )



            st.write(

                f"**Gender:** {gender}"

            )



            st.write(

                f"**Country:** {country}"

            )



            st.write(

                f"**Occupation:** {occupation}"

            )





        with s2:



            bmi = (

                weight_kg

                / ((height_cm / 100) ** 2)

            )



            st.write(

                f"**BMI:** {bmi:.2f}"

            )



            st.write(

                f"**Height:** {height_cm:.0f} cm"

            )



            st.write(

                f"**Weight:** {weight_kg:.1f} kg"

            )



            st.write(

                f"**Pulse Pressure:** "

                f"{ap_hi - ap_lo:.0f} mmHg"

            )





        with s3:



            st.write(

                f"**Systolic BP:** {ap_hi}"

            )



            st.write(

                f"**Diastolic BP:** {ap_lo}"

            )



            st.write(

                f"**Cholesterol:** "

                f"{cholesterol_label}"

            )



            st.write(

                f"**Glucose:** "

                f"{glucose_label}"

            )





        st.warning(

            "Academic demonstration only. This system "

            "should not be used to diagnose, treat, or "

            "make medical decisions for real patients."

        )





# ============================================================

# MODEL PERFORMANCE

# ============================================================

elif page == "📊 Model Performance":



    st.subheader(

        "📊 Model Performance"

    )



    st.caption(

        "Comparison of the four classification "

        "algorithms evaluated in the project."

    )





    metrics_df = pd.DataFrame(

        {

            "Model": [

                "Logistic Regression",

                "Decision Tree",

                "Random Forest",

                "Linear SVM",

            ],



            "Accuracy": [

                72.53,

                72.86,

                73.12,

                72.47,

            ],



            "Precision": [

                75.06,

                75.15,

                75.14,

                75.40,

            ],



            "Recall": [

                67.44,

                68.27,

                69.05,

                66.65,

            ],



            "F1 Score": [

                71.04,

                71.55,

                71.97,

                70.76,

            ],

        }

    )





    render_table(metrics_df.style.format(

            {

                "Accuracy": "{:.2f}%",

                "Precision": "{:.2f}%",

                "Recall": "{:.2f}%",

                "F1 Score": "{:.2f}%",

            }

        ))





    st.markdown("---")



    c1, c2 = st.columns(2)





    with c1:



        st.subheader(

            "🎯 Accuracy Comparison"

        )



        chart_df = (

            metrics_df[

                [

                    "Model",

                    "Accuracy"

                ]

            ]

            .set_index("Model")

        )



        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(chart_df.index, chart_df["Accuracy"])
        ax.set_title("Model Accuracy Comparison")
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        plt.xticks(rotation=20)
        for bar, value in zip(bars, chart_df["Accuracy"]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.2f}%", ha="center", va="bottom")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)





    with c2:



        st.subheader(

            "📈 F1 Score Comparison"

        )



        f1_df = (
            metrics_df[["Model", "F1 Score"]]
            .set_index("Model")
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(f1_df.index, f1_df["F1 Score"])
        ax.set_title("Model F1 Score Comparison")
        ax.set_xlabel("Model")
        ax.set_ylabel("F1 Score (%)")
        ax.set_ylim(0, 100)
        plt.xticks(rotation=20)

        for bar, value in zip(bars, f1_df["F1 Score"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1,
                f"{value:.2f}%",
                ha="center",
                va="bottom"
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)





    st.markdown("---")



    st.subheader(

        "🌲 Random Forest Feature Importance"

    )





    importance_df = (

        get_feature_importance_table(

            model

        )

    )





    if (

        importance_df is not None

        and not importance_df.empty

    ):



        chart = importance_df.sort_values("Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(chart["Feature"], chart["Importance"])
        ax.set_title("Random Forest Feature Importance")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        for bar, value in zip(bars, chart["Importance"]):
            ax.text(value, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", ha="left")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        importance_display = importance_df.copy()
        importance_display["Importance"] = importance_display["Importance"].map(lambda x: f"{x:.4f}")
        render_table(importance_display)



        st.caption(

            "Feature importance indicates predictive "

            "contribution within the model; it does "

            "not establish causation."

        )



    else:



        st.info(

            "Feature importance could not be extracted "

            "automatically from the saved model."

        )





# ============================================================

# DATA EXPLORER

# ============================================================

elif page == "📁 Data Explorer":



    st.subheader(

        "📁 Data Explorer"

    )





    c1, c2, c3 = st.columns(3)





    with c1:



        st.metric(

            "Rows",

            f"{df.shape[0]:,}"

        )





    with c2:



        st.metric(

            "Columns",

            df.shape[1]

        )





    with c3:



        st.metric(

            "Missing Values",

            f"{int(df.isna().sum().sum()):,}"

        )





    st.markdown("---")



    st.subheader(

        "🔎 Dataset Preview"

    )





    max_rows = min(

        100,

        max(5, len(df))

    )



    n_rows = st.slider(

        "Rows to display",

        5,

        max_rows,

        min(20, max_rows)

    )





    render_table(df.head(n_rows))





    st.markdown("---")



    left, right = st.columns(2)





    with left:



        st.subheader(

            "📐 Numeric Summary"

        )



        numeric_cols = (

            df

            .select_dtypes(

                include=np.number

            )

            .columns

        )





        if len(numeric_cols):



            render_table(df[numeric_cols]

                .describe()

                .T

                .round(2))





    with right:



        st.subheader(

            "⚠️ Missing Values"

        )



        missing = (

            df

            .isna()

            .sum()

            .sort_values(

                ascending=False

            )

        )



        missing = missing[

            missing > 0

        ]





        if len(missing):



            render_table(missing.rename(

                    "Missing Count"

                ).to_frame())



        else:



            st.success(

                "No missing values detected."

            )





    if target_exists:



        st.markdown("---")



        st.subheader(

            "🎯 Target Distribution"

        )



        target_counts = (

            df["disease"]

            .value_counts()

            .sort_index()

        )



        target_labels = {0: "No Disease (0)", 1: "Disease (1)"}
        target_display = target_counts.rename(index=lambda x: target_labels.get(x, str(x)))
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(target_display.index, target_display.values)
        ax.set_title("Target Distribution")
        ax.set_xlabel("Disease Class")
        ax.set_ylabel("Number of Records")
        for bar, value in zip(bars, target_display.values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{int(value):,}", ha="center", va="bottom")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)





# ============================================================

# ABOUT

# ============================================================

else:



    st.subheader(

        "ℹ️ About the Project"

    )





    st.markdown(

        """

        ### ❤️ Heart Disease Detection



        This project is a **binary classification machine-learning

        system** designed to predict whether a record belongs to

        the disease class or the non-disease class.



        ### 🧠 Machine Learning Pipeline



        **Dataset → EDA → Preprocessing → Feature Engineering

        → Model Training → Model Evaluation → Random Forest

        → Streamlit UI**



        ### 🔧 Main Technologies



        - Python

        - Pandas

        - NumPy

        - Scikit-learn

        - Joblib

        - Streamlit



        ### 🤖 Models Compared



        1. Logistic Regression

        2. Decision Tree

        3. Random Forest

        4. Linear SVM



        ### ⭐ Final Model



        Random Forest was selected as the final model because

        it achieved the strongest overall performance among

        the tested models in this project.



        ### 🧮 Engineered Features



        - **Age in years:** age converted from days using 365.25

        - **BMI:** weight / height²

        - **Pulse pressure:** systolic BP − diastolic BP



        ### ⚠️ Important



        This is an academic machine-learning demonstration.

        Predictions should not be interpreted as medical diagnoses

        or used as a substitute for professional medical advice.

        """

    )





    st.markdown("---")



    st.subheader(

        "🔄 How Prediction Works"

    )





    st.info(

        "Patient Inputs  →  Feature Engineering  →  "

        "Preprocessing Pipeline  →  Random Forest  →  "

        "Class Prediction + Probability"

    )





# ============================================================

# FOOTER

# ============================================================

st.markdown(

    """

    <div class="footer">

        Heart Disease Detection • Machine Learning Academic Project • Streamlit

    </div>

    """,

    unsafe_allow_html=True,

)

