# dashboard/app.py
"""
Tableau de bord enseignant - Suivi des performances scolaires.
Interface intuitive, langage pédagogique, aide à la décision.
Utilise l'API FastAPI déployée.
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ------------------------------------------------------------
# Configuration de la page
# ------------------------------------------------------------
st.set_page_config(
    page_title="EduPredict - Suivi des Élèves",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Styles CSS personnalisés (identité visuelle sobre et claire)
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
        padding: 2.5rem 2rem;
        border-radius: 30px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: "";
        position: absolute;
        top: -30%;
        right: -5%;
        width: 350px;
        height: 350px;
        background: rgba(255,255,255,0.03);
        border-radius: 50%;
    }
    .main-header h1 {
        font-size: 3.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        letter-spacing: -1px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
        margin: 0.3rem 0;
    }

    .card {
        background: white;
        border-radius: 20px;
        padding: 1.8rem 1.5rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
        transition: transform 0.2s, box-shadow 0.2s;
        margin-bottom: 1.5rem;
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.08);
    }

    .risk-high {
        background: linear-gradient(135deg, #dc3545, #c82333);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 24px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(220,53,69,0.3);
        animation: pulse 2s infinite;
    }
    .risk-moderate {
        background: linear-gradient(135deg, #fd7e14, #e96b0c);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 24px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(253,126,20,0.3);
    }
    .risk-low {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 24px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(40,167,69,0.3);
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }

    .resource-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.04);
        border-left: 8px solid #2C5364;
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .resource-card:hover {
        transform: translateX(8px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    }
    .resource-card h4 {
        color: #0F2027;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }

    .divider {
        height: 2px;
        background: linear-gradient(90deg, #0F2027, #2C5364, transparent);
        margin: 2.5rem 0;
    }

    .footer {
        text-align: center;
        padding: 2rem;
        color: #7f8c8d;
        font-size: 0.85rem;
        border-top: 1px solid #ecf0f1;
    }

    .stButton>button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.2s !important;
        background: linear-gradient(135deg, #0F2027, #2C5364) !important;
        color: white !important;
        border: none !important;
    }
    .stButton>button:hover {
        transform: scale(1.03) !important;
        box-shadow: 0 8px 20px rgba(15,32,39,0.4) !important;
        color: white !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 12px 28px;
        border-radius: 20px 20px 0 0;
        background-color: #f1f3f5;
        transition: all 0.3s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #dee2e6;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0F2027, #2C5364) !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# URL de l'API
# ------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

# ------------------------------------------------------------
# Initialisation de la session
# ------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame()
if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = None
if "current_input" not in st.session_state:
    st.session_state.current_input = None

# ------------------------------------------------------------
# En-tête principal
# ------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🎓 EduPredict</h1>
        <p>Plateforme intelligente de suivi des performances scolaires</p>
        <p style="font-size: 0.9rem; opacity: 0.7;">Régression linéaire • IA explicable • Pilotage pédagogique</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Barre latérale
# ------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=80)
    st.markdown("## ⚙️ Configuration")
    st.markdown("L'application utilise l'API déployée pour les prédictions.")
    st.markdown(f"**Endpoint :** `{API_URL}`")
    
    # Statistiques de session si historique
    if not st.session_state.history.empty:
        st.markdown("---")
        st.markdown("### 📊 Statistiques de session")
        n_total = len(st.session_state.history)
        n_high = len(st.session_state.history[st.session_state.history["Score prédit"] < 60])
        n_mod = len(st.session_state.history[
            (st.session_state.history["Score prédit"] >= 60) & 
            (st.session_state.history["Score prédit"] < 70)
        ])
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total", n_total)
        with col_b:
            st.metric("🔴 Risque", n_high)
        with col_c:
            st.metric("🟠 Modéré", n_mod)
    
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #7f8c8d;">
        <strong>EduPredict v1.0</strong><br>
        Modèle : Régression linéaire<br>
        MAE : 0.45 point • R² : 0.77
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# Onglets principaux
# ------------------------------------------------------------
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 VUE D'ENSEMBLE",
    "📋 PRÉDICTION",
    "📁 ANALYSE PAR LOT",
    "📊 FACTEURS DE RISQUE",
    "📚 RESSOURCES"
])

# ============================================================
# ONGLET 0 : Vue d'ensemble
# ============================================================
with tab0:
    st.markdown("## 🏠 Vue d'ensemble — Année scolaire 2024-25")
    st.markdown("*Synthèse des performances prédites et des alertes. Les données présentées sont fictives et servent d'illustration.*")

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.metric("Élèves suivis", "247", delta="6 classes")
    with col_m2:
        st.metric("Score moyen", "67.2 / 100", delta="+2.1 pts")
    with col_m3:
        st.metric("Risque élevé", "41", delta="16.6%")
    with col_m4:
        st.metric("Risque modéré", "68", delta="27.5%")
    with col_m5:
        st.metric("Bon niveau", "138", delta="55.9%")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("### Distribution des scores par classe")
        classes = ['5e A', '4e B', '3e B', '2nde A', '1ère C', 'Tle S']
        scores_moyens = [63.2, 65.4, 57.8, 70.1, 66.3, 69.4]
        fig = px.bar(x=classes, y=scores_moyens, color=classes,
                     color_discrete_sequence=['#2C5364']*len(classes),
                     labels={'x':'Classe', 'y':'Score moyen'},
                     title="Score moyen prédit par classe")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
    with col_g2:
        st.markdown("### Répartition des niveaux de risque")
        labels = ['Risque élevé', 'Risque modéré', 'Bon niveau']
        values = [41, 68, 138]
        colors = ['#dc3545', '#fd7e14', '#28a745']
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6, marker_colors=colors)])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📈 Évolution trimestrielle")
    trimestres = ['T1', 'T2', 'T3']
    scores_trim = [63.2, 65.1, 67.2]
    fig = px.line(x=trimestres, y=scores_trim, markers=True,
                  labels={'x':'Trimestre', 'y':'Score moyen'},
                  title="Progression du score moyen")
    fig.update_traces(line_color='#2C5364', line_width=3)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# ONGLET 1 : Prédiction individuelle
# ============================================================
with tab1:
    st.markdown("## 📋 Prédiction individuelle")
    st.markdown("*Saisissez les informations de l'élève pour estimer sa note à l'examen. Tous les champs sont importants.*")
    
    with st.form("student_form", clear_on_submit=False):
        st.markdown("### 📖 Scolarité")
        col1, col2, col3 = st.columns(3)
        with col1:
            hours = st.number_input("Heures d'étude par semaine", 0, 60, 20, help="Nombre d'heures consacrées aux devoirs et révisions.")
            attendance = st.slider("Taux de présence (%)", 0, 100, 80, help="Pourcentage de cours effectivement suivis.")
            prev_scores = st.number_input("Score antérieur", 0, 100, 70, help="Dernière note connue ou moyenne des évaluations précédentes.")
        with col2:
            tutoring = st.number_input("Séances de tutorat", 0, 15, 1, help="Nombre de séances de soutien scolaire suivies.")
            sleep = st.number_input("Heures de sommeil par nuit", 4, 12, 7, help="Moyenne d'heures de sommeil.")
            physical = st.number_input("Activité physique (h/sem)", 0, 15, 3, help="Heures d'activité physique par semaine.")
        with col3:
            extracurricular = st.selectbox("Activités extrascolaires", ["Oui", "Non"], help="Participation à des clubs, sports, musique...")
            internet = st.selectbox("Accès Internet à domicile", ["Oui", "Non"])
            school_type = st.selectbox("Type d'établissement", ["Public", "Privé"])
        
        st.markdown("### 👨‍👩‍👧 Environnement familial")
        col4, col5, col6 = st.columns(3)
        with col4:
            parental = st.selectbox("Implication parentale", ["Faible", "Moyenne", "Élevée"], help="Suivi des devoirs, participation aux réunions...")
            family_income = st.selectbox("Revenu familial", ["Faible", "Moyen", "Élevé"])
            parent_edu = st.selectbox("Éducation des parents", ["Lycée", "Université", "Post-universitaire"], help="Plus haut diplôme obtenu par les parents.")
        with col5:
            access = st.selectbox("Accès aux ressources", ["Faible", "Moyen", "Bon"], help="Livres, ordinateur, espace de travail calme.")
            distance = st.selectbox("Distance domicile-école", ["Proche", "Modérée", "Loin"])
            gender = st.selectbox("Genre", ["Garçon", "Fille"])
        with col6:
            motivation = st.selectbox("Motivation", ["Faible", "Moyenne", "Élevée"], help="Envie de réussir, persévérance.")
            teacher_quality = st.selectbox("Qualité perçue de l'enseignant", ["Faible", "Moyenne", "Bonne"])
            peer = st.selectbox("Influence des camarades", ["Négative", "Neutre", "Positive"], help="Les amis poussent-ils à travailler ou au contraire ?")
        
        st.markdown("### 🩺 Santé")
        learning_dis = st.selectbox("Troubles de l'apprentissage diagnostiqués", ["Non", "Oui"], help="Dyslexie, TDAH, etc.")
        
        # Conversion vers les valeurs attendues par l'API (anglais)
        submitted = st.form_submit_button("🔍 Estimer le score", use_container_width=True)
    
    if submitted:
        # Mapper les labels français vers les labels anglais
        map_label = {
            "Oui": "Yes", "Non": "No",
            "Faible": "Low", "Moyenne": "Medium", "Moyen": "Medium", "Élevée": "High", "Élevé": "High",
            "Bonne": "High", "Bon": "High",
            "Proche": "Near", "Modérée": "Moderate", "Loin": "Far",
            "Garçon": "Male", "Fille": "Female",
            "Négative": "Negative", "Neutre": "Neutral", "Positive": "Positive",
            "Public": "Public", "Privé": "Private",
            "Lycée": "High School", "Université": "College", "Post-universitaire": "Postgraduate"
        }
        input_data = {
            "Hours_Studied": hours,
            "Attendance": attendance,
            "Parental_Involvement": map_label[parental],
            "Access_to_Resources": map_label[access],
            "Extracurricular_Activities": map_label[extracurricular],
            "Sleep_Hours": sleep,
            "Previous_Scores": prev_scores,
            "Motivation_Level": map_label[motivation],
            "Internet_Access": map_label[internet],
            "Tutoring_Sessions": tutoring,
            "Family_Income": map_label[family_income],
            "Teacher_Quality": map_label[teacher_quality],
            "School_Type": map_label[school_type],
            "Peer_Influence": map_label[peer],
            "Physical_Activity": physical,
            "Learning_Disabilities": map_label[learning_dis],
            "Parental_Education_Level": map_label[parent_edu],
            "Distance_from_Home": map_label[distance],
            "Gender": map_label[gender],
        }
        
        try:
            response = requests.post(API_URL, json=input_data, timeout=5)
            if response.status_code == 200:
                score = response.json()["Exam_Score_predicted"]
            else:
                st.error(f"Erreur API : {response.status_code}")
                score = None
        except requests.exceptions.ConnectionError:
            st.error("❌ Impossible de se connecter à l'API. Veuillez vérifier qu'elle est en ligne.")
            score = None
        
        if score is not None:
            st.session_state.current_prediction = score
            st.session_state.current_input = input_data
            
            # Historique
            history_row = input_data.copy()
            history_row["Score prédit"] = round(score, 2)
            history_row["Date"] = datetime.now().strftime("%d/%m/%Y %H:%M")
            if st.session_state.history.empty:
                st.session_state.history = pd.DataFrame([history_row])
            else:
                st.session_state.history = pd.concat(
                    [st.session_state.history, pd.DataFrame([history_row])],
                    ignore_index=True,
                )
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("## 📊 Résultat de l'estimation")
            
            if score < 60:
                risk_class = "risk-high"
                risk_emoji, risk_label, risk_desc = "🔴", "RISQUE ÉLEVÉ", "Cet élève a besoin d'un accompagnement renforcé."
            elif score < 70:
                risk_class = "risk-moderate"
                risk_emoji, risk_label, risk_desc = "🟠", "RISQUE MODÉRÉ", "Une surveillance et des encouragements sont recommandés."
            else:
                risk_class = "risk-low"
                risk_emoji, risk_label, risk_desc = "🟢", "BON NIVEAU", "L'élève est sur une trajectoire positive."
            
            col_res1, col_res2 = st.columns([1, 1])
            with col_res1:
                st.markdown(
                    f"""
                    <div class="{risk_class}">
                        <h2>{risk_emoji} {score:.1f} / 100</h2>
                        <h3>{risk_label}</h3>
                        <p>{risk_desc}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Score estimé", f"{score:.1f} / 100")
                with col_m2:
                    gap = max(0, 70 - score)
                    st.metric("Objectif 70/100", f"{gap:.1f} points à rattraper" if gap > 0 else "✅ Atteint")
                with col_m3:
                    status = "🔴 Prioritaire" if score < 60 else ("🟠 À surveiller" if score < 70 else "🟢 Satisfaisant")
                    st.metric("Statut", status)
            
            with col_res2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=score,
                    number={'suffix': " / 100", 'font': {'size': 40}},
                    title={'text': "Score estimé", 'font': {'size': 20}},
                    delta={'reference': 67.24, 'increasing': {'color': "green"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "rgba(15,32,39,0.9)", 'thickness': 0.2},
                        'steps': [
                            {'range': [0, 60], 'color': '#ff6b6b'},
                            {'range': [60, 70], 'color': '#ffa502'},
                            {'range': [70, 100], 'color': '#26de81'}],
                        'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.8, 'value': score}
                    }
                ))
                fig.update_layout(height=350, margin=dict(l=30, r=30, t=50, b=30))
                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# ONGLET 2 : Analyse par lot
# ============================================================
with tab2:
    st.markdown("## 📁 Analyse par lot")
    st.markdown("*Importez un fichier CSV contenant plusieurs élèves pour obtenir leurs scores estimés en une seule fois.*")
    
    st.markdown("### 📥 Modèle de fichier à télécharger")
    example_data = {
        "Heures d'étude": [20, 15, 30],
        "Présence (%)": [80, 70, 95],
        "Implication parentale": ["Moyenne", "Faible", "Élevée"],
        "Accès aux ressources": ["Moyen", "Faible", "Bon"],
        "Activités extrascolaires": ["Oui", "Non", "Oui"],
        "Heures de sommeil": [7, 6, 8],
        "Score antérieur": [70, 60, 85],
        "Motivation": ["Moyenne", "Faible", "Élevée"],
        "Accès Internet": ["Oui", "Oui", "Oui"],
        "Séances de tutorat": [1, 0, 3],
        "Revenu familial": ["Moyen", "Faible", "Élevé"],
        "Qualité enseignant": ["Moyenne", "Faible", "Bonne"],
        "Type d'école": ["Public", "Public", "Privé"],
        "Influence des pairs": ["Positive", "Négative", "Positive"],
        "Activité physique": [3, 2, 5],
        "Troubles apprentissage": ["Non", "Non", "Non"],
        "Éducation des parents": ["Lycée", "Lycée", "Post-universitaire"],
        "Distance domicile": ["Proche", "Modérée", "Proche"],
        "Genre": ["Garçon", "Fille", "Garçon"]
    }
    template_df = pd.DataFrame(example_data)
    csv_template = template_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📄 Télécharger le modèle CSV (avec exemples)",
        csv_template,
        "modele_eleves.csv",
        "text/csv",
        help="Cliquez pour télécharger un fichier CSV que vous pourrez remplir avec vos propres élèves."
    )
    
    st.markdown("---")
    uploaded_file = st.file_uploader("📤 Déposer votre fichier CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ {len(batch_df)} élèves chargés.")
            with st.expander("👀 Aperçu des premières lignes"):
                st.dataframe(batch_df.head(10), use_container_width=True)
            
            # Conversion des colonnes françaises vers l'anglais pour l'API
            col_map = {
                "Heures d'étude": "Hours_Studied",
                "Présence (%)": "Attendance",
                "Implication parentale": "Parental_Involvement",
                "Accès aux ressources": "Access_to_Resources",
                "Activités extrascolaires": "Extracurricular_Activities",
                "Heures de sommeil": "Sleep_Hours",
                "Score antérieur": "Previous_Scores",
                "Motivation": "Motivation_Level",
                "Accès Internet": "Internet_Access",
                "Séances de tutorat": "Tutoring_Sessions",
                "Revenu familial": "Family_Income",
                "Qualité enseignant": "Teacher_Quality",
                "Type d'école": "School_Type",
                "Influence des pairs": "Peer_Influence",
                "Activité physique": "Physical_Activity",
                "Troubles apprentissage": "Learning_Disabilities",
                "Éducation des parents": "Parental_Education_Level",
                "Distance domicile": "Distance_from_Home",
                "Genre": "Gender"
            }
            # Renommer si les colonnes sont en français (sinon garder tel quel)
            if set(col_map.keys()).issubset(set(batch_df.columns)):
                batch_df.rename(columns=col_map, inplace=True)
            # Convertir les valeurs françaises en anglais
            value_map = {
                "Oui": "Yes", "Non": "No",
                "Faible": "Low", "Moyenne": "Medium", "Moyen": "Medium", "Élevée": "High", "Élevé": "High",
                "Bonne": "High", "Bon": "High",
                "Proche": "Near", "Modérée": "Moderate", "Loin": "Far",
                "Garçon": "Male", "Fille": "Female",
                "Négative": "Negative", "Neutre": "Neutral", "Positive": "Positive",
                "Public": "Public", "Privé": "Private",
                "Lycée": "High School", "Université": "College", "Post-universitaire": "Postgraduate"
            }
            for col in batch_df.columns:
                if batch_df[col].dtype == object:
                    batch_df[col] = batch_df[col].map(lambda x: value_map.get(str(x), x))
            
            if st.button("🚀 Lancer l'estimation pour tous les élèves", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                predictions = []
                total = len(batch_df)
                
                for idx, (_, row) in enumerate(batch_df.iterrows()):
                    input_row = row.to_dict()
                    try:
                        resp = requests.post(API_URL, json=input_row, timeout=5)
                        if resp.status_code == 200:
                            pred = resp.json()["Exam_Score_predicted"]
                        else:
                            pred = None
                    except:
                        pred = None
                    
                    predictions.append(pred)
                    progress_bar.progress((idx + 1) / total)
                    status_text.text(f"En cours : {idx + 1}/{total} élèves...")
                
                batch_df["Score estimé"] = predictions
                batch_df["Niveau de risque"] = batch_df["Score estimé"].apply(
                    lambda x: "🔴 Élevé" if pd.notna(x) and x < 60 else ("🟠 Modéré" if pd.notna(x) and x < 70 else ("🟢 Bon" if pd.notna(x) else "⚠️ Erreur"))
                )
                
                status_text.text("✅ Estimation terminée !")
                
                st.markdown("### 📊 Résumé")
                valid = batch_df["Score estimé"].notna()
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    st.metric("Total élèves", len(batch_df))
                with col_s2:
                    st.metric("🔴 Risque élevé", len(batch_df[valid & (batch_df["Score estimé"] < 60)]))
                with col_s3:
                    st.metric("🟠 Modéré", len(batch_df[valid & (batch_df["Score estimé"] >= 60) & (batch_df["Score estimé"] < 70)]))
                with col_s4:
                    st.metric("🟢 Bon niveau", len(batch_df[valid & (batch_df["Score estimé"] >= 70)]))
                
                if valid.any():
                    fig_dist = px.histogram(
                        batch_df[valid], x="Score estimé", nbins=20,
                        title="Répartition des scores estimés",
                        color_discrete_sequence=['#2C5364'],
                        labels={"Score estimé": "Score", "count": "Nombre d'élèves"}
                    )
                    fig_dist.add_vline(x=60, line_dash="dash", line_color="red", annotation_text="Seuil risque")
                    fig_dist.add_vline(x=70, line_dash="dash", line_color="orange", annotation_text="Seuil modéré")
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                st.dataframe(batch_df, use_container_width=True)
                
                csv_result = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Télécharger les résultats complets",
                    csv_result,
                    f"resultats_lot_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")

# ============================================================
# ONGLET 3 : Facteurs clés – Ce qui compte vraiment
# ============================================================
with tab3:
    st.markdown("## 📊 Qu'est-ce qui influence vraiment le score ?")
    st.markdown(
        "Notre modèle a identifié les trois facteurs sur lesquels vous pouvez agir "
        "pour aider un élève à progresser. Les chiffres donnent une idée de l'impact, "
        "mais chaque situation reste unique."
    )

    # ----- Les 3 leviers principaux (coefficients arrondis) -----
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="card" style="text-align:center;">
                <h3 style="color:#0F2027;">📅 Assiduité</h3>
                <p style="font-size:2rem; font-weight:700; color:#2C5364;">+2,3 points</p>
                <p style="font-size:0.9rem; color:#555;">
                    Chaque point de présence en plus <strong>augmente</strong> le score d'environ 2,3 points.
                    C'est le levier le plus puissant.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="card" style="text-align:center;">
                <h3 style="color:#0F2027;">📚 Heures d'étude</h3>
                <p style="font-size:2rem; font-weight:700; color:#2C5364;">+1,7 point</p>
                <p style="font-size:0.9rem; color:#555;">
                    Une heure d'étude hebdomadaire supplémentaire <strong>augmente</strong> 
                    le score d'environ 1,7 point.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="card" style="text-align:center;">
                <h3 style="color:#0F2027;">🤝 Entourage positif</h3>
                <p style="font-size:2rem; font-weight:700; color:#2C5364;">+1,1 point</p>
                <p style="font-size:0.9rem; color:#555;">
                    Passer d'un entourage négatif à positif <strong>augmente</strong> 
                    le score d'environ 1,1 point.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "💡 **À retenir** : l'assiduité, le temps d'étude et un environnement social favorable "
        "sont les trois piliers de la réussite scolaire selon nos données. "
        "Les autres facteurs (motivation, accès aux ressources, etc.) jouent aussi un rôle, "
        "mais leur impact individuel est moins marqué."
    )

    # Historique des prédictions (inchangé)
    if not st.session_state.history.empty:
        st.markdown("---")
        st.markdown("### 📜 Historique de vos estimations")
        st.dataframe(
            st.session_state.history.sort_values("Date", ascending=False),
            use_container_width=True,
        )
        csv_hist = st.session_state.history.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Télécharger l'historique",
            csv_hist,
            f"historique_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
        )

# ============================================================
# ONGLET 4 : Ressources & Recommandations
# ============================================================
with tab4:
    st.markdown("## 📚 Ressources et conseils personnalisés")
    st.markdown("*Des pistes d'action concrètes pour aider chaque élève à progresser.*")
    
    if st.session_state.current_prediction is not None:
        score = st.session_state.current_prediction
        inp = st.session_state.current_input  # contient les labels anglais
        
        st.markdown("### 🎯 Conseils pour cet élève")
        st.markdown(f"*Score estimé : {score:.1f}/100*")
        
        recs = []
        # On réutilise les mêmes seuils mais avec des messages plus explicites
        if inp["Attendance"] < 70:
            recs.append({
                "titre": "Améliorer l'assiduité",
                "desc": "L'assiduité est le levier le plus puissant. Mettre en place un suivi hebdomadaire, contacter la famille, valoriser chaque progrès.",
                "lien": "https://www.education.gouv.fr/la-lutte-contre-le-decrochage-scolaire-7214"
            })
        if inp["Hours_Studied"] < 15:
            recs.append({
                "titre": "Augmenter le temps d'étude",
                "desc": "Fixer un planning de révision régulier, proposer un accompagnement aux devoirs.",
                "lien": "https://www.pass-ton-bacsn.com/"
            })
        if inp["Access_to_Resources"] == "Low":
            recs.append({
                "titre": "Faciliter l'accès aux ressources",
                "desc": "Orienter vers les bibliothèques, les plateformes gratuites (Khan Academy, Afterclasse).",
                "lien": "https://www.khanacademy.org/"
            })
        if inp["Tutoring_Sessions"] < 2 and score < 70:
            recs.append({
                "titre": "Renforcer le tutorat",
                "desc": "Proposer des séances de soutien supplémentaires, individuelles ou en petits groupes.",
                "lien": "https://www.superprof.sn/"
            })
        if inp["Learning_Disabilities"] == "Yes":
            recs.append({
                "titre": "Accompagnement spécialisé",
                "desc": "Consulter un orthophoniste, un psychologue scolaire ou un enseignant référent pour un suivi adapté.",
                "lien": "https://www.tdah-france.fr/"
            })
        if inp["Motivation_Level"] == "Low":
            recs.append({
                "titre": "Stimuler la motivation",
                "desc": "Fixer des objectifs atteignables, valoriser les réussites, encourager le mentorat par les pairs.",
                "lien": "https://www.reseau-canope.fr/"
            })
        if inp["Internet_Access"] == "No":
            recs.append({
                "titre": "Garantir un accès numérique",
                "desc": "Orienter vers les espaces publics numériques, les médiathèques, ou les associations fournissant du matériel.",
                "lien": "https://www.orange.com/fr/engagements/education-numerique"
            })
        if score < 60:
            recs.append({
                "titre": "Mettre en place un programme de remédiation prioritaire",
                "desc": "Tutorat quotidien, suivi psychologique, lien renforcé avec la famille. Une intervention rapide est nécessaire.",
                "lien": "https://www.education.gouv.fr/bo/21/Hebdo37/MENE2127642C.htm"
            })
        
        if recs:
            for rec in recs:
                st.markdown(
                    f"""
                    <div class="resource-card">
                        <h4>{rec['titre']}</h4>
                        <p>{rec['desc']}</p>
                        <a href="{rec['lien']}" target="_blank" style="color: #2C5364; text-decoration: none; font-weight: 500;">🔗 Ressource utile →</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.success("✅ Aucun facteur de risque majeur détecté. Continuez à encourager l'élève dans ses efforts.")
    
    st.markdown("---")
    st.markdown("### 🌐 Ressources générales pour les enseignants")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("#### 📖 Plateformes de révision")
        st.markdown("""
        - [Pass Ton Bac SN](https://www.pass-ton-bacsn.com/)
        - [Khan Academy](https://fr.khanacademy.org/)
        - [Afterclasse](https://www.afterclasse.fr/)
        - [Maxicours](https://www.maxicours.com/)
        - [Les Bons Profs](https://www.lesbonsprofs.com/)
        """)
        st.markdown("#### 👨‍🏫 Tutorat et soutien")
        st.markdown("""
        - [Superprof](https://www.superprof.sn/)
        - [Acadomia](https://www.acadomia.fr/)
        - [Complétude](https://www.completude.com/)
        - [Apprentus](https://www.apprentus.com/)
        """)
    with col_r2:
        st.markdown("#### 🧠 Bien-être et santé")
        st.markdown("""
        - [Fil Santé Jeunes](https://www.filsantejeunes.com/)
        - [Nightline](https://www.nightline.fr/)
        - [e-Enfance](https://www.e-enfance.org/)
        """)
        st.markdown("#### 📱 Applications utiles")
        st.markdown("""
        - Forest (concentration)
        - Quizlet (flashcards)
        - Notion (organisation)
        - Duolingo (langues)
        """)
        st.markdown("#### 🏫 Institutions")
        st.markdown("""
        - [Ministère Éducation Sénégal](https://www.education.sn/)
        - [UNESCO Éducation](https://fr.unesco.org/themes/education)
        """)

# ------------------------------------------------------------
# Pied de page
# ------------------------------------------------------------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="footer">
        <p>🎓 <strong>EduPredict v1.0</strong> — Projet REG06 — Suivi des Performances Scolaires</p>
        <p>Développé pour accompagner les équipes pédagogiques · Machine Learning · Streamlit · FastAPI</p>
    </div>
    """,
    unsafe_allow_html=True,
)