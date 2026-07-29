# ==========================================
# IMPORT LIBRARIES
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
import plotly.express as px

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Medical Question Classification",
    page_icon="🩺",
    layout="wide"
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("medquad.csv")

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("medquad_model.pkl")

tfidf = joblib.load("tfidf_vectorizer.pkl")

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main{
background:#f5f7fa;
}

h1{
color:#0066cc;
font-weight:bold;
}

h2{
color:#0d47a1;
}

h3{
color:#1565c0;
}

.stButton>button{

background:#1976d2;
color:white;
font-size:18px;
font-weight:bold;
border-radius:10px;
height:50px;
width:100%;

}

.stButton>button:hover{

background:#0d47a1;
color:white;

}

.card{

background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 0px 15px rgba(0,0,0,0.15);

}

.footer{

text-align:center;
font-size:16px;
color:gray;

}

</style>

""",unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.image(
"https://img.icons8.com/color/512/stethoscope.png",
width=120
)

st.sidebar.title("Medical NLP")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dataset",
        "📈 Visualization",
        "🔍 Prediction",
        "📌 Dashboard",
        "⚙ Workflow",
        "ℹ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info("""

Medical Question Classification

Algorithm :
Logistic Regression

Feature Extraction :
TF-IDF

Dataset :
MedQuAD

""")

# ==========================================
# HOME PAGE
# ==========================================

if menu=="🏠 Home":

    st.title("🩺 Medical Question Classification")

    st.write("")

    st.markdown("""

This application predicts the medical category of a user question using

**Natural Language Processing (NLP)** and **Machine Learning**.

""")

    st.write("")

    col1,col2,col3=st.columns(3)

    with col1:

        st.metric(
        "Dataset Size",
        len(df)
        )

    with col2:

        st.metric(
        "Categories",
        df["focus_area"].nunique()
        )

    with col3:

        st.metric(
        "Algorithm",
        "Logistic Regression"
        )

    st.write("---")

    st.subheader("Project Features")

    st.success("✔ Medical Question Classification")

    st.success("✔ TF-IDF Feature Extraction")

    st.success("✔ Logistic Regression Model")

    st.success("✔ Irrelevant Question Detection")

    st.success("✔ Interactive Dashboard")

    st.success("✔ Data Visualization")

    st.write("---")

    st.subheader("Sample Medical Questions")

    st.info("What are the symptoms of diabetes?")

    st.info("How is asthma treated?")

    st.info("What causes migraine headaches?")

    st.info("What are the symptoms of heart disease?")

    st.info("How can kidney stones be prevented?")

    st.write("---")

    st.subheader("Project Workflow")

    st.code("""

Medical Question

        ↓

Text Cleaning

        ↓

Tokenization

        ↓

TF-IDF Vectorization

        ↓

Logistic Regression

        ↓

Predicted Category

""")

    st.write("---")

    st.markdown("""

<div class="card">

<h2>Objective</h2>

<p>

The objective of this project is to classify medical questions into the
appropriate medical focus area using Natural Language Processing.

</p>

</div>

""",unsafe_allow_html=True)

    st.write("")

    st.balloons()
# ==========================================
# DATASET PAGE
# ==========================================

elif menu == "📊 Dataset":

    st.title("📊 MedQuAD Dataset Explorer")

    st.write("")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Dataset",
            "Statistics",
            "Search",
            "Download"
        ]
    )

    # ==========================
    # TAB 1
    # ==========================

    with tab1:

        st.subheader("Dataset Preview")

        rows = st.slider(
            "Select Number of Rows",
            5,
            100,
            10
        )

        st.dataframe(
            df.head(rows),
            use_container_width=True
        )

        st.write("")

        st.subheader("Dataset Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Rows", df.shape[0])

        with c2:
            st.metric("Columns", df.shape[1])

        with c3:
            st.metric(
                "Categories",
                df["focus_area"].nunique()
            )

    # ==========================
    # TAB 2
    # ==========================

    with tab2:

        st.subheader("Dataset Statistics")

        st.dataframe(
            df.describe(include="all"),
            use_container_width=True
        )

        st.write("")

        st.subheader("Missing Values")

        missing = df.isnull().sum().reset_index()

        missing.columns = [
            "Column",
            "Missing Values"
        ]

        st.dataframe(
            missing,
            use_container_width=True
        )

        st.write("")

        st.subheader("Column Names")

        st.write(list(df.columns))

    # ==========================
    # TAB 3
    # ==========================

    with tab3:

        st.subheader("Search Medical Questions")

        keyword = st.text_input(
            "Enter Keyword"
        )

        if keyword != "":

            result = df[
                df["question"].str.contains(
                    keyword,
                    case=False,
                    na=False
                )
            ]

            st.write(
                "Matching Records:",
                len(result)
            )

            st.dataframe(
                result,
                use_container_width=True
            )

        else:

            st.info(
                "Enter a keyword to search."
            )

    # ==========================
    # TAB 4
    # ==========================

    with tab4:

        st.subheader("Download Dataset")

        csv = df.to_csv(
            index=False
        )

        st.download_button(

            label="Download CSV",

            data=csv,

            file_name="medquad.csv",

            mime="text/csv"

        )

# ==========================================
# VISUALIZATION PAGE
# ==========================================

elif menu == "📈 Visualization":

    st.title("📈 Dataset Visualization")

    st.write("")

    st.subheader("Top 10 Medical Categories")

    top10 = (
        df["focus_area"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    top10.columns = [
        "Category",
        "Count"
    ]

    fig = px.bar(

        top10,

        x="Category",

        y="Count",

        text="Count",

        color="Count"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.write("---")

    st.subheader("Top 10 Categories Pie Chart")

    fig2 = px.pie(

        top10,

        names="Category",

        values="Count"

    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.write("---")

    st.subheader("Dataset Distribution")

    st.bar_chart(
        df["focus_area"].value_counts().head(20)
    )

    st.write("---")

    st.subheader("Random Medical Questions")

    sample = df.sample(10)

    st.dataframe(
        sample[
            [
                "question",
                "focus_area"
            ]
        ],
        use_container_width=True
    )

    st.write("---")

    st.subheader("Top Categories")

    st.table(
        top10
    )

    st.write("---")

    st.success(
        "Visualization Loaded Successfully."
    )
# ==========================================
# PREDICTION PAGE
# ==========================================

elif menu == "🔍 Prediction":

    st.title("🔍 Medical Question Classification")

    st.write("")

    st.markdown("""
    Enter your medical question below.
    The model will classify it into the appropriate medical category.
    If the question is unrelated to medicine, it will display
    **Irrelevant Question**.
    """)

    st.write("")

    # -----------------------------
    # Medical Keywords
    # -----------------------------

    medical_keywords = [

        "disease","symptom","symptoms","medicine","medication",
        "doctor","hospital","treatment","therapy","patient",
        "diabetes","cancer","heart","kidney","brain","liver",
        "blood","infection","virus","bacteria","covid",
        "fever","headache","pain","fracture","arthritis",
        "thyroid","stroke","pregnancy","asthma","allergy",
        "skin","eye","ear","nose","throat","lungs",
        "vomiting","nausea","injury","operation","surgery",
        "tablet","capsule","antibiotic","vaccine","health"

    ]

    def is_medical(text):

        text = text.lower()

        for word in medical_keywords:

            if word in text:

                return True

        return False

    # -----------------------------
    # User Input
    # -----------------------------

    question = st.text_area(

        "Enter Medical Question",

        height=180,

        placeholder="Example : What are the symptoms of diabetes?"

    )

    st.write("")

    c1,c2,c3 = st.columns([1,2,1])

    with c2:

        predict = st.button(

            "Predict",

            use_container_width=True

        )

    # -----------------------------
    # Prediction
    # -----------------------------

    if predict:

        if question.strip()=="":

            st.warning("Please enter a question.")

        else:

            if not is_medical(question):

                st.error("❌ Irrelevant Question")

            else:

                with st.spinner("Predicting..."):

                    cleaned = re.sub(

                        r'[^a-zA-Z\s]',

                        '',

                        question.lower()

                    )

                    vector = tfidf.transform([cleaned])

                    prediction = model.predict(vector)[0]

                    confidence = None

                    # Logistic Regression supports predict_proba
                    if hasattr(model, "predict_proba"):

                        probs = model.predict_proba(vector)

                        confidence = np.max(probs)

                st.success("Prediction Completed Successfully")

                st.write("")

                st.markdown(f"""

                <div style="background:#E8F5E9;
                padding:20px;
                border-radius:12px;
                border-left:8px solid green;">

                <h2>Predicted Category</h2>

                <h3>{prediction}</h3>

                </div>

                """,unsafe_allow_html=True)

                st.write("")

                if confidence is not None:

                    st.progress(float(confidence))

                    st.info(
                        f"Confidence : {confidence*100:.2f}%"
                    )

                    if confidence < 0.40:

                        st.warning(
                            "Low confidence prediction."
                        )

                st.write("---")

                st.subheader("Original Question")

                st.info(question)

                st.write("---")

                st.subheader("Processed Question")

                st.code(cleaned)

                st.write("---")

                st.subheader("Prediction Summary")

                summary = pd.DataFrame({

                    "Question":[question],

                    "Prediction":[prediction],

                    "Confidence":[
                        f"{confidence*100:.2f}%"
                        if confidence is not None
                        else "Not Available"
                    ]

                })

                st.dataframe(

                    summary,

                    use_container_width=True

                )

    st.write("---")

    st.subheader("Sample Questions")

    samples=[

        "What are the symptoms of diabetes?",

        "How is asthma treated?",

        "What causes migraine?",

        "How to reduce blood pressure?",

        "What are kidney stones?",

        "What are symptoms of breast cancer?"

    ]

    for q in samples:

        st.success(q)

    st.write("---")

    st.subheader("Try Irrelevant Questions")

    st.error("Who won the IPL yesterday?")

    st.error("Tell me a joke")

    st.error("What is Python programming?")

    st.error("Best movie in 2026?")

    st.write("---")

    st.subheader("Model Information")

    col1,col2,col3=st.columns(3)

    with col1:

        st.metric(

            "Algorithm",

            "Logistic Regression"

        )

    with col2:

        st.metric(

            "Vectorizer",

            "TF-IDF"

        )

    with col3:

        st.metric(

            "Dataset",

            "MedQuAD"

        )

    st.write("---")

    st.success("Medical Question Classification Completed Successfully ✅")
elif menu == "📌 Dashboard":

    st.title("📌 Admin Dashboard")

    st.write("")

    col1,col2,col3,col4=st.columns(4)

    with col1:
        st.metric(
            "Total Records",
            len(df)
        )

    with col2:
        st.metric(
            "Medical Categories",
            df["focus_area"].nunique()
        )

    with col3:
        st.metric(
            "Model",
            "Logistic Regression"
        )

    with col4:
        st.metric(
            "Vectorizer",
            "TF-IDF"
        )

    st.write("---")

    st.subheader("Dataset Overview")

    st.dataframe(df.head(20),use_container_width=True)

    st.write("---")

    st.subheader("Top 20 Categories")

    top20=df["focus_area"].value_counts().head(20)

    st.bar_chart(top20)

    st.write("---")

    st.subheader("Dataset Distribution")

    chart=px.pie(
        values=top20.values,
        names=top20.index,
        title="Top Categories"
    )

    st.plotly_chart(chart,use_container_width=True)

    st.write("---")

    st.success("Dashboard Loaded Successfully")
# ==========================================
# WORKFLOW PAGE
# ==========================================

elif menu == "⚙ Workflow":

    st.title("⚙ Project Workflow")

    st.write("")

    st.markdown("""
    ## Medical Question Classification Workflow
    """)

    st.write("---")

    st.subheader("Step 1 : Dataset Collection")

    st.success("""
✔ Dataset : MedQuAD

✔ Medical Questions

✔ Medical Categories (Focus Area)
""")

    st.write("---")

    st.subheader("Step 2 : Data Preprocessing")

    st.info("""

• Remove Special Characters

• Convert to Lowercase

• Remove Extra Spaces

• Stopword Removal

• Lemmatization

""")

    st.write("---")

    st.subheader("Step 3 : Feature Extraction")

    st.success("""

TF-IDF converts text into numerical vectors.

Machine Learning algorithms use these vectors
for training and prediction.

""")

    st.write("---")

    st.subheader("Step 4 : Train Test Split")

    st.code("""

80% → Training Data

20% → Testing Data

""")

    st.write("---")

    st.subheader("Step 5 : Model Training")

    st.success("""

Algorithm Used

✔ Logistic Regression

""")

    st.write("---")

    st.subheader("Step 6 : Prediction")

    st.code("""

User Question

        ↓

Text Cleaning

        ↓

TF-IDF Vectorization

        ↓

Logistic Regression

        ↓

Medical Category

""")

    st.write("---")

    st.subheader("Overall Flow")

    st.code("""

Medical Question

↓

Preprocessing

↓

TF-IDF

↓

Logistic Regression

↓

Prediction

""")

# ==========================================
# ABOUT PAGE
# ==========================================

elif menu == "ℹ About":

    st.title("ℹ About Project")

    st.write("")

    st.markdown("""

## Medical Question Classification using NLP

This application predicts the medical category
of user questions using Natural Language Processing
and Machine Learning.

""")

    st.write("---")

    col1,col2=st.columns(2)

    with col1:

        st.subheader("Project Details")

        st.write("""

✔ Domain : Healthcare

✔ Dataset : MedQuAD

✔ NLP : TF-IDF

✔ Algorithm : Logistic Regression

✔ Language : Python

✔ Framework : Streamlit

""")

    with col2:

        st.subheader("Libraries Used")

        st.write("""

✔ Pandas

✔ NumPy

✔ Streamlit

✔ Scikit-Learn

✔ Joblib

✔ Plotly

✔ Matplotlib

""")

    st.write("---")

    st.subheader("Project Objectives")

    st.success("""

✔ Classify Medical Questions

✔ Improve Medical Information Retrieval

✔ Demonstrate NLP Techniques

✔ Fast Medical Category Prediction

""")

    st.write("---")

    st.subheader("Advantages")

    st.info("""

• User Friendly

• Fast Prediction

• Easy to Use

• Automatic Classification

• Professional Dashboard

""")

    st.write("---")

    st.subheader("Future Enhancements")

    st.write("""

• BERT Model

• LSTM Model

• Voice Assistant

• Medical Chatbot

• Disease Recommendation

• Cloud Deployment

""")

    st.write("---")

    st.subheader("Technology Stack")

    tech = pd.DataFrame({

        "Technology":[

            "Python",

            "Streamlit",

            "TF-IDF",

            "Logistic Regression",

            "Pandas",

            "Scikit-Learn"

        ],

        "Purpose":[

            "Programming",

            "Web Application",

            "Feature Extraction",

            "Classification",

            "Data Handling",

            "Machine Learning"

        ]

    })

    st.dataframe(
        tech,
        use_container_width=True
    )

    st.write("---")

    st.subheader("Performance")

    st.metric(
        "Prediction",
        "Medical Category"
    )

    st.metric(
        "Dataset",
        "MedQuAD"
    )

    st.metric(
        "Algorithm",
        "Logistic Regression"
    )

    st.write("---")

    st.markdown("""

<div style="background:#1565c0;
padding:25px;
border-radius:15px;
color:white;
text-align:center;">

<h2>Medical Question Classification</h2>

<h4>Natural Language Processing Project</h4>

<p>Developed using Python, Streamlit & Machine Learning</p>

</div>

""",unsafe_allow_html=True)

# ==========================================
# CONTACT SECTION
# ==========================================

st.write("")
st.write("---")

st.subheader("📞 Contact")

st.info("""

Project : Medical Question Classification

Technology : NLP + Machine Learning

Framework : Streamlit

Dataset : MedQuAD

""")

# ==========================================
# FOOTER
# ==========================================

st.write("---")

st.markdown("""

<div style="text-align:center;
padding:20px;
color:gray;">

© 2026 Medical Question Classification using NLP

Developed with ❤️ using Python, Streamlit and Scikit-Learn

</div>

""",unsafe_allow_html=True)
# ==========================================
# ADMIN DASHBOARD
# ==========================================
