# dashboard/app.py
"""
Tableau de bord enseignant - Suivi des performances scolaires.
Fonctionnalités : prédiction individuelle, analyse par lot, facteurs de risque, recommandations.
Utilise l'API FastAPI déployée.
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

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
# Styles CSS personnalisés
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
        padding: 3rem 2rem;
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
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }

    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
        margin: 0.3rem 0;
    }

    .card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
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
        border-radius: 12px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.2s;
        background: linear-gradient(135deg, #0F2027, #2C5364);
        color: white;
        border: none;
    }

    .stButton>button:hover {
        transform: scale(1.03);
        box-shadow: 0 8px 20px rgba(15,32,39,0.4);
        color: white;
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

    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        color: #6c757d;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
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
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 PRÉDICTION INDIVIDUELLE",
    "📁 ANALYSE PAR LOT",
    "📊 FACTEURS DE RISQUE",
    "📚 RESSOURCES & RECOMMANDATIONS"
])

# ============================================================
# ONGLET 1 : Prédiction Individuelle
# ============================================================
with tab1:
    st.markdown("## 📋 Saisie des caractéristiques de l'élève")
    st.markdown("*Remplissez les champs ci-dessous pour obtenir une prédiction personnalisée.*")
    
    with st.form("student_form", clear_on_submit=False):
        # Sections thématiques avec info-bulles
        st.markdown("### 📖 Profil Académique")
        col1, col2, col3 = st.columns(3)
        with col1:
            hours = st.number_input("📚 Heures d'étude / semaine", 0, 60, 20,
                                    help="Nombre d'heures consacrées aux révisions et devoirs.")
            attendance = st.slider("📅 Taux de présence (%)", 0, 100, 80,
                                   help="Pourcentage de cours suivis.")
            prev_scores = st.number_input("📝 Score antérieur", 0, 100, 70,
                                          help="Dernier résultat connu ou moyenne des examens précédents.")
        with col2:
            tutoring = st.number_input("👨‍🏫 Séances de tutorat", 0, 15, 1,
                                       help="Nombre de séances de soutien scolaire suivies.")
            sleep = st.number_input("😴 Heures de sommeil", 4, 12, 7,
                                    help="Nombre moyen d'heures de sommeil par nuit.")
            physical = st.number_input("🏃 Activité physique (h/sem)", 0, 15, 3,
                                       help="Heures d'activité physique par semaine.")
        with col3:
            extracurricular = st.selectbox("🎭 Activités extrascolaires", ["No", "Yes"],
                                           help="Participation à des activités en dehors des cours (sport, musique, etc.)")
            internet = st.selectbox("🌐 Accès Internet", ["No", "Yes"],
                                    help="Dispose d'une connexion Internet à domicile.")
            school_type = st.selectbox("🏫 Type d'établissement", ["Public", "Private"],
                                       help="Établissement public ou privé.")
        
        st.markdown("### 👨‍👩‍👧 Environnement Socio-Familial")
        col4, col5, col6 = st.columns(3)
        with col4:
            parental = st.selectbox("👪 Implication parentale", ["Low", "Medium", "High"],
                                    help="Niveau d'implication des parents dans le suivi scolaire.")
            family_income = st.selectbox("💰 Revenu familial", ["Low", "Medium", "High"],
                                         help="Niveau de revenu du foyer.")
            parent_edu = st.selectbox("🎓 Éducation des parents", 
                                      ["High School", "College", "Postgraduate"],
                                      help="Plus haut diplôme obtenu par les parents.")
        with col5:
            access = st.selectbox("📦 Accès aux ressources", ["Low", "Medium", "High"],
                                  help="Disponibilité de livres, ordinateur, espace de travail calme.")
            distance = st.selectbox("📍 Distance domicile-école", ["Near", "Moderate", "Far"],
                                    help="Temps de trajet jusqu'à l'établissement.")
            gender = st.selectbox("⚧ Genre", ["Male", "Female"])
        with col6:
            motivation = st.selectbox("🔥 Niveau de motivation", ["Low", "Medium", "High"],
                                      help="Auto-évaluation de la motivation à réussir.")
            teacher_quality = st.selectbox("👩‍🏫 Qualité perçue de l'enseignant", ["Low", "Medium", "High"],
                                           help="Perception de la qualité de l'enseignement.")
            peer = st.selectbox("🤝 Influence des pairs", ["Negative", "Neutral", "Positive"],
                                help="Influence des camarades sur le comportement scolaire.")
        
        st.markdown("### 🩺 Santé & Apprentissage")
        learning_dis = st.selectbox("🧠 Troubles de l'apprentissage diagnostiqués", ["No", "Yes"],
                                    help="Présence de troubles comme la dyslexie, TDAH, etc.")
        
        submitted = st.form_submit_button("🔍 Analyser le profil", use_container_width=True)
    
    if submitted:
        input_data = {
            "Hours_Studied": hours, "Attendance": attendance,
            "Parental_Involvement": parental, "Access_to_Resources": access,
            "Extracurricular_Activities": extracurricular, "Sleep_Hours": sleep,
            "Previous_Scores": prev_scores, "Motivation_Level": motivation,
            "Internet_Access": internet, "Tutoring_Sessions": tutoring,
            "Family_Income": family_income, "Teacher_Quality": teacher_quality,
            "School_Type": school_type, "Peer_Influence": peer,
            "Physical_Activity": physical, "Learning_Disabilities": learning_dis,
            "Parental_Education_Level": parent_edu, "Distance_from_Home": distance,
            "Gender": gender,
        }
        
        # Appel API
        try:
            response = requests.post(API_URL, json=input_data, timeout=5)
            if response.status_code == 200:
                score = response.json()["Exam_Score_predicted"]
            else:
                st.error(f"Erreur API : {response.status_code}")
                score = None
        except requests.exceptions.ConnectionError:
            st.error("❌ Impossible de se connecter à l'API. Vérifiez qu'elle est en ligne.")
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
            
            # Résultats
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("## 📊 Résultat de l'Analyse")
            
            if score < 60:
                risk_class, risk_emoji, risk_label, risk_desc = (
                    "risk-high", "🔴", "RISQUE ÉLEVÉ", "Cet élève nécessite une intervention prioritaire."
                )
            elif score < 70:
                risk_class, risk_emoji, risk_label, risk_desc = (
                    "risk-moderate", "🟠", "RISQUE MODÉRÉ", "Une attention particulière est recommandée."
                )
            else:
                risk_class, risk_emoji, risk_label, risk_desc = (
                    "risk-low", "🟢", "BON NIVEAU", "L'élève est sur une trajectoire positive."
                )
            
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
                
                st.markdown("### 📈 Métriques détaillées")
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Score prédit", f"{score:.1f} / 100")
                with col_m2:
                    gap = max(0, 70 - score)
                    st.metric("Points pour 70/100", f"{gap:.1f}" if gap > 0 else "✅ Atteint")
                with col_m3:
                    status = "🔴 Prioritaire" if score < 60 else ("🟠 Surveillance" if score < 70 else "🟢 Satisfaisant")
                    st.metric("Statut", status)
            
            with col_res2:
                # Jauge interactive
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=score,
                    number={'suffix': " / 100", 'font': {'size': 45}},
                    title={'text': "Score Prédit", 'font': {'size': 22}},
                    delta={'reference': 67.24, 'increasing': {'color': "green"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                        'bar': {'color': "rgba(15,32,39,0.9)", 'thickness': 0.2},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#ccc",
                        'steps': [
                            {'range': [0, 60], 'color': '#ff6b6b'},
                            {'range': [60, 70], 'color': '#ffa502'},
                            {'range': [70, 100], 'color': '#26de81'}],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.8,
                            'value': score}
                    }
                ))
                fig.update_layout(height=400, margin=dict(l=30, r=30, t=50, b=30))
                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# ONGLET 2 : Analyse par Lot
# ============================================================
with tab2:
    st.markdown("## 📁 Analyse par lot")
    st.markdown("*Importez un fichier CSV contenant les caractéristiques de plusieurs élèves.*")
    
    # Template téléchargeable avec exemple
    st.markdown("### 📥 Télécharger un template")
    example_data = {
        "Hours_Studied": [20, 15, 30],
        "Attendance": [80, 70, 95],
        "Parental_Involvement": ["Medium", "Low", "High"],
        "Access_to_Resources": ["Medium", "Low", "High"],
        "Extracurricular_Activities": ["Yes", "No", "Yes"],
        "Sleep_Hours": [7, 6, 8],
        "Previous_Scores": [70, 60, 85],
        "Motivation_Level": ["Medium", "Low", "High"],
        "Internet_Access": ["Yes", "Yes", "Yes"],
        "Tutoring_Sessions": [1, 0, 3],
        "Family_Income": ["Medium", "Low", "High"],
        "Teacher_Quality": ["Medium", "Low", "High"],
        "School_Type": ["Public", "Public", "Private"],
        "Peer_Influence": ["Positive", "Negative", "Positive"],
        "Physical_Activity": [3, 2, 5],
        "Learning_Disabilities": ["No", "No", "No"],
        "Parental_Education_Level": ["High School", "High School", "Postgraduate"],
        "Distance_from_Home": ["Near", "Moderate", "Near"],
        "Gender": ["Male", "Female", "Male"]
    }
    template_df = pd.DataFrame(example_data)
    csv_template = template_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📄 Télécharger le template CSV (avec exemples)",
        csv_template,
        "template_eleves.csv",
        "text/csv",
        help="Fichier CSV pré-rempli avec 3 élèves fictifs pour vous aider à comprendre le format."
    )
    
    st.markdown("---")
    uploaded_file = st.file_uploader("📤 Importer un fichier CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ {len(batch_df)} élèves chargés.")
            
            with st.expander("👀 Aperçu des données"):
                st.dataframe(batch_df.head(10), use_container_width=True)
            
            if st.button("🚀 Lancer les prédictions", type="primary", use_container_width=True):
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
                            pred = np.nan
                    except:
                        pred = np.nan
                    
                    predictions.append(round(pred, 2) if not np.isnan(pred) else None)
                    progress_bar.progress((idx + 1) / total)
                    status_text.text(f"Traitement : {idx + 1}/{total} élèves...")
                
                batch_df["Score prédit"] = predictions
                batch_df["Niveau de risque"] = batch_df["Score prédit"].apply(
                    lambda x: "🔴 Élevé" if pd.notna(x) and x < 60 else ("🟠 Modéré" if pd.notna(x) and x < 70 else ("🟢 Faible" if pd.notna(x) else "⚠️ Erreur"))
                )
                
                status_text.text("✅ Prédictions terminées !")
                
                # Résumé
                st.markdown("### 📊 Résumé des résultats")
                valid = batch_df["Score prédit"].notna()
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    st.metric("Total élèves", len(batch_df))
                with col_s2:
                    st.metric("🔴 Risque élevé", len(batch_df[valid & (batch_df["Score prédit"] < 60)]))
                with col_s3:
                    st.metric("🟠 Modéré", len(batch_df[valid & (batch_df["Score prédit"] >= 60) & (batch_df["Score prédit"] < 70)]))
                with col_s4:
                    st.metric("🟢 Bon niveau", len(batch_df[valid & (batch_df["Score prédit"] >= 70)]))
                
                # Distribution
                if valid.any():
                    fig_dist = px.histogram(
                        batch_df[valid], x="Score prédit", nbins=20,
                        title="Distribution des scores prédits",
                        color_discrete_sequence=['#2C5364'],
                        labels={"Score prédit": "Score", "count": "Nombre d'élèves"}
                    )
                    fig_dist.add_vline(x=60, line_dash="dash", line_color="red", annotation_text="Seuil risque")
                    fig_dist.add_vline(x=70, line_dash="dash", line_color="orange", annotation_text="Seuil modéré")
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                st.dataframe(batch_df, use_container_width=True)
                
                csv_result = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Télécharger les résultats complets",
                    csv_result,
                    f"predictions_lot_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")

# ============================================================
# ONGLET 3 : Facteurs de Risque
# ============================================================
with tab3:
    st.markdown("## 📊 Analyse des Facteurs de Risque")
    st.markdown("*Visualisez l'impact de chaque variable sur la performance scolaire.*")
    
    # Chargement des coefficients depuis l'API (ou cache)
    @st.cache_data
    def get_coefficients():
        # Nous n'avons pas d'endpoint pour les coefs, mais nous pouvons les embarquer depuis le notebook
        # Ici on les définit en dur (issus du notebook final)
        coefficients = {
            'Attendance': 2.307045,
            'Hours_Studied': 1.745091,
            'Peer_Influence': 1.081664,  # ordinal
            'Internet_Access_Yes': 1.072856,
            'Access_to_Resources': 1.054157,
            'Parental_Involvement': 0.995126,
            'Learning_Disabilities_Yes': -0.844337,
            'Previous_Scores': 0.724497,
            'Tutoring_Sessions': 0.631927,
            'Extracurricular_Activities_Yes': 0.594917,
            'Teacher_Quality': 0.543349,
            'Motivation_Level': 0.534964,
            'Family_Income': 0.502724,
            'Distance_from_Home': -0.496451,
            'Parental_Education_Level': 0.494408,
            'Physical_Activity': 0.184673,
            'Gender_Male': -0.033454,
            'Sleep_Hours': -0.028378,
            'School_Type_Public': 0.028317
        }
        return pd.DataFrame(list(coefficients.items()), columns=['Variable', 'Coefficient'])

    coef_df = get_coefficients()
    coef_df['Abs'] = coef_df['Coefficient'].abs()
    coef_df['Impact'] = coef_df['Coefficient'].apply(lambda x: "✅ Positif" if x > 0 else "❌ Négatif")
    coef_df = coef_df.sort_values('Abs', ascending=False)
    
    col_top1, col_top2 = st.columns([2, 1])
    with col_top1:
        st.markdown("### 🔝 Top 15 des Variables les Plus Influentes")
        fig_bar = px.bar(
            coef_df.head(15).sort_values('Coefficient'),
            x='Coefficient', y='Variable', orientation='h',
            color='Impact',
            color_discrete_map={"✅ Positif": "#28a745", "❌ Négatif": "#dc3545"},
            title="Impact des variables sur le score prédit"
        )
        fig_bar.add_vline(x=0, line_width=1, line_color="black")
        fig_bar.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col_top2:
        st.markdown("### 📋 Détail des Coefficients")
        st.dataframe(
            coef_df.head(10)[['Variable', 'Coefficient', 'Impact']].set_index('Variable').style.format({
                'Coefficient': '{:.3f}'
            }),
            use_container_width=True,
        )
    
    st.markdown("### 💡 Interprétation")
    st.markdown("""
    - **Coefficient positif** : augmente le score prédit quand la valeur augmente.
    - **Coefficient négatif** : diminue le score prédit.
    - **Valeur absolue** : importance de la variable.
    """)
    
    # Graphique radar des forces/faiblesses (pour un élève type)
    st.markdown("### 🎯 Profil type d'un élève")
    # Sélection des variables les plus importantes
    top_vars = coef_df.head(10)['Variable'].tolist()
    # Pour illustrer, on prend un élève fictif avec des valeurs moyennes
    radar_data = {
        'Attendance': 80,
        'Hours_Studied': 20,
        'Peer_Influence': 1,  # ordinal 0=Négatif, 1=Neutre, 2=Positif
        'Internet_Access': 1,
        'Access_to_Resources': 1,
        'Parental_Involvement': 1,
        'Learning_Disabilities': 0,
        'Previous_Scores': 70,
        'Tutoring_Sessions': 1,
        'Extracurricular_Activities': 1
    }
    # Normalisation pour radar (sur une échelle 0-100)
    max_vals = {
        'Attendance': 100,
        'Hours_Studied': 40,
        'Peer_Influence': 2,
        'Internet_Access': 1,
        'Access_to_Resources': 2,
        'Parental_Involvement': 2,
        'Learning_Disabilities': 1,  # inversé plus bas
        'Previous_Scores': 100,
        'Tutoring_Sessions': 8,
        'Extracurricular_Activities': 1
    }
    categories = top_vars
    values = []
    for var in categories:
        if var in radar_data:
            norm = radar_data[var] / max_vals[var] * 100
            if var == 'Learning_Disabilities_Yes':
                norm = 100 - norm  # inverser car négatif
            values.append(norm)
        else:
            values.append(50)  # défaut
    
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Élève moyen',
        line_color='#2C5364'
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=False,
        title="Profil de l'élève moyen (points forts / points faibles)"
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Historique
    if not st.session_state.history.empty:
        st.markdown("---")
        st.markdown("### 📜 Historique des Prédictions")
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
    st.markdown("## 📚 Ressources Pédagogiques & Recommandations")
    st.markdown("*Des outils pour accompagner chaque élève vers la réussite.*")
    
    if st.session_state.current_prediction is not None:
        score = st.session_state.current_prediction
        inp = st.session_state.current_input
        
        st.markdown("### 🎯 Recommandations Personnalisées")
        st.markdown(f"*Basées sur le profil de l'élève (score prédit : {score:.1f}/100)*")
        
        recs = []
        if inp["Attendance"] < 70:
            recs.append({
                "titre": "📅 Améliorer l'assiduité",
                "desc": "L'assiduité est le facteur le plus déterminant. Mettre en place un suivi hebdomadaire.",
                "lien": "https://www.education.gouv.fr/la-lutte-contre-le-decrochage-scolaire-7214"
            })
        if inp["Hours_Studied"] < 15:
            recs.append({
                "titre": "📚 Augmenter le temps d'étude",
                "desc": "Planifier des sessions d'étude régulières avec un planning structuré.",
                "lien": "https://www.pass-ton-bacsn.com/"
            })
        if inp["Access_to_Resources"] == "Low":
            recs.append({
                "titre": "📦 Améliorer l'accès aux ressources",
                "desc": "Orienter vers des bibliothèques, ressources en ligne gratuites et associations.",
                "lien": "https://www.khanacademy.org/"
            })
        if inp["Tutoring_Sessions"] < 2 and score < 70:
            recs.append({
                "titre": "👨‍🏫 Renforcer le tutorat",
                "desc": "Proposer des séances de soutien scolaire supplémentaires.",
                "lien": "https://www.superprof.sn/"
            })
        if inp["Learning_Disabilities"] == "Yes":
            recs.append({
                "titre": "🧠 Accompagnement spécialisé",
                "desc": "Consulter un orthophoniste ou psychologue scolaire pour un suivi adapté.",
                "lien": "https://www.tdah-france.fr/"
            })
        if inp["Motivation_Level"] == "Low":
            recs.append({
                "titre": "🔥 Stimuler la motivation",
                "desc": "Fixer des objectifs atteignables, valoriser les progrès, mentorat par les pairs.",
                "lien": "https://www.reseau-canope.fr/"
            })
        if inp["Internet_Access"] == "No":
            recs.append({
                "titre": "🌐 Faciliter l'accès numérique",
                "desc": "Orienter vers les espaces numériques publics, les médiathèques connectées.",
                "lien": "https://www.orange.com/fr/engagements/education-numerique"
            })
        if score < 60:
            recs.append({
                "titre": "🚨 Programme de Remédiation Prioritaire",
                "desc": "Accompagnement intensif : tutorat quotidien, suivi psychologique, contact famille.",
                "lien": "https://www.education.gouv.fr/bo/21/Hebdo37/MENE2127642C.htm"
            })
        
        if recs:
            for rec in recs[:6]:
                st.markdown(
                    f"""
                    <div class="resource-card">
                        <h4>{rec['titre']}</h4>
                        <p>{rec['desc']}</p>
                        <a href="{rec['lien']}" target="_blank" style="color: #2C5364; text-decoration: none; font-weight: 500;">🔗 En savoir plus →</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.success("✅ Aucun facteur de risque majeur détecté. Continuez à encourager l'élève.")
    
    # Ressources générales
    st.markdown("---")
    st.markdown("### 🌐 Ressources Générales")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("#### 📖 Plateformes de Révision")
        st.markdown("""
        - [Pass Ton Bac SN](https://www.pass-ton-bacsn.com/)
        - [Khan Academy](https://fr.khanacademy.org/)
        - [Afterclasse](https://www.afterclasse.fr/)
        - [Maxicours](https://www.maxicours.com/)
        - [Les Bons Profs](https://www.lesbonsprofs.com/)
        """)
        st.markdown("#### 👨‍🏫 Tutorat & Soutien")
        st.markdown("""
        - [Superprof](https://www.superprof.sn/)
        - [Acadomia](https://www.acadomia.fr/)
        - [Complétude](https://www.completude.com/)
        - [Apprentus](https://www.apprentus.com/)
        """)
    with col_r2:
        st.markdown("#### 🧠 Bien-être & Santé")
        st.markdown("""
        - [Fil Santé Jeunes](https://www.filsantejeunes.com/)
        - [Nightline](https://www.nightline.fr/)
        - [e-Enfance](https://www.e-enfance.org/)
        """)
        st.markdown("#### 📱 Applications Utiles")
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