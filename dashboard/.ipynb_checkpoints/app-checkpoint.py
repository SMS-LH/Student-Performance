# dashboard/app.py
"""
Tableau de bord enseignant pour le suivi des performances scolaires.
Fonctionnalités : prédiction individuelle, analyse par lot, facteurs de risque, recommandations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import joblib
from pathlib import Path
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
# Styles CSS personnalisés (identité visuelle "grande école")
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem 2rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 12px 40px rgba(30, 60, 114, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    .metric-card {
        background: white;
        border-radius: 20px;
        padding: 1.8rem 1.5rem;
        box-shadow: 0 6px 24px rgba(0,0,0,0.04);
        text-align: center;
        border: 1px solid #f0f4ff;
        transition: transform 0.25s, box-shadow 0.25s;
    }
    
    .metric-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 16px 32px rgba(0,0,0,0.08);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 2.5rem;
        border-radius: 24px;
        text-align: center;
        box-shadow: 0 12px 30px rgba(231, 76, 60, 0.3);
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: white;
        padding: 2.5rem;
        border-radius: 24px;
        text-align: center;
        box-shadow: 0 12px 30px rgba(243, 156, 18, 0.3);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white;
        padding: 2.5rem;
        border-radius: 24px;
        text-align: center;
        box-shadow: 0 12px 30px rgba(39, 174, 96, 0.3);
    }
    
    .resource-card {
        background: white;
        border-radius: 20px;
        padding: 1.8rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.04);
        border-left: 6px solid #2a5298;
        margin-bottom: 1.2rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .resource-card:hover {
        transform: translateX(8px);
        box-shadow: 0 10px 24px rgba(0,0,0,0.08);
    }
    
    .resource-card h4 {
        color: #1e3c72;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .divider {
        height: 2px;
        background: linear-gradient(90deg, #1e3c72, #2a5298, transparent);
        margin: 2.5rem 0;
    }
    
    .footer {
        text-align: center;
        padding: 2.5rem;
        color: #7f8c8d;
        font-size: 0.85rem;
        border-top: 1px solid #ecf0f1;
    }
    
    /* Boutons stylisés */
    .stButton>button {
        border-radius: 12px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 16px rgba(42, 82, 152, 0.3);
    }
    
    /* Onglets */
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 12px 28px;
        border-radius: 16px 16px 0 0;
        background-color: #f8f9fa;
        transition: all 0.3s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3c72, #2a5298) !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Chemins
# ------------------------------------------------------------
DASHBOARD_DIR = Path(__file__).parent
HOME_DIR = DASHBOARD_DIR.parent
MODEL_PATH = HOME_DIR / "models" / "final_pipeline.pkl"
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
        <p style="font-size: 0.85rem; opacity: 0.75;">Régression linéaire • IA explicable • Pilotage pédagogique</p>
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
    
    mode = st.radio(
        "Mode de fonctionnement",
        options=["🌐 Appeler l'API FastAPI", "💻 Utiliser le modèle local"],
        help="L'API doit être lancée avec `uvicorn api.app:app`",
    )
    
    st.markdown("---")
    
    # Statistiques de session si historique
    if not st.session_state.history.empty:
        st.markdown("### 📊 Statistiques de session")
        n_total = len(st.session_state.history)
        n_high_risk = len(st.session_state.history[st.session_state.history["Score prédit"] < 60])
        n_moderate = len(st.session_state.history[
            (st.session_state.history["Score prédit"] >= 60) & 
            (st.session_state.history["Score prédit"] < 70)
        ])
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total", n_total)
        with col_b:
            st.metric("🔴 Risque élevé", n_high_risk)
        with col_c:
            st.metric("🟠 Modéré", n_moderate)
    
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

# Chargement conditionnel du modèle local
if mode == "💻 Utiliser le modèle local":
    if not MODEL_PATH.exists():
        st.error(f"⚠️ Modèle introuvable dans `{MODEL_PATH}`")
        pipeline = None
    else:
        pipeline = joblib.load(MODEL_PATH)
else:
    pipeline = None

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
        # Sections thématiques
        st.markdown("### 📖 Profil Académique")
        col1, col2, col3 = st.columns(3)
        with col1:
            hours = st.number_input("📚 Heures d'étude / semaine", 0, 60, 20)
            attendance = st.slider("📅 Taux de présence (%)", 0, 100, 80)
            prev_scores = st.number_input("📝 Score antérieur", 0, 100, 70)
        with col2:
            tutoring = st.number_input("👨‍🏫 Séances de tutorat", 0, 15, 1)
            sleep = st.number_input("😴 Heures de sommeil", 4, 12, 7)
            physical = st.number_input("🏃 Activité physique (h/sem)", 0, 15, 3)
        with col3:
            extracurricular = st.selectbox("🎭 Activités extrascolaires", ["No", "Yes"])
            internet = st.selectbox("🌐 Accès Internet", ["No", "Yes"])
            school_type = st.selectbox("🏫 Type d'établissement", ["Public", "Private"])
        
        st.markdown("### 👨‍👩‍👧 Environnement Socio-Familial")
        col4, col5, col6 = st.columns(3)
        with col4:
            parental = st.selectbox("👪 Implication parentale", ["Low", "Medium", "High"])
            family_income = st.selectbox("💰 Revenu familial", ["Low", "Medium", "High"])
            parent_edu = st.selectbox("🎓 Éducation des parents", 
                                      ["High School", "College", "Postgraduate"])
        with col5:
            access = st.selectbox("📦 Accès aux ressources", ["Low", "Medium", "High"])
            distance = st.selectbox("📍 Distance domicile-école", ["Near", "Moderate", "Far"])
            gender = st.selectbox("⚧ Genre", ["Male", "Female"])
        with col6:
            motivation = st.selectbox("🔥 Niveau de motivation", ["Low", "Medium", "High"])
            teacher_quality = st.selectbox("👩‍🏫 Qualité perçue de l'enseignant", ["Low", "Medium", "High"])
            peer = st.selectbox("🤝 Influence des pairs", ["Negative", "Neutral", "Positive"])
        
        st.markdown("### 🩺 Santé & Apprentissage")
        learning_dis = st.selectbox("🧠 Troubles de l'apprentissage diagnostiqués", ["No", "Yes"])
        
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
        
        # Prédiction via API ou modèle local
        score = None
        if mode == "🌐 Appeler l'API FastAPI":
            try:
                response = requests.post(API_URL, json=input_data, timeout=5)
                if response.status_code == 200:
                    score = response.json()["Exam_Score_predicted"]
                else:
                    st.error(f"Erreur API : {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Impossible de se connecter à l'API. Lancez-la avec `uvicorn api.app:app`.")
        else:
            if pipeline is None:
                st.error("❌ Modèle local non disponible.")
            else:
                df_input = pd.DataFrame([input_data])
                score = pipeline.predict(df_input)[0]
        
        if score is not None:
            st.session_state.current_prediction = score
            st.session_state.current_input = input_data
            
            # Ajout à l'historique
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
            
            # Affichage des résultats
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
                # Jauge Plotly
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=score,
                    number={'suffix': " / 100", 'font': {'size': 40}},
                    title={'text': "Score Prédit", 'font': {'size': 20}},
                    delta={'reference': 67.24, 'increasing': {'color': "green"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "darkblue", 'thickness': 0.15},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 60], 'color': '#e74c3c'},
                            {'range': [60, 70], 'color': '#f39c12'},
                            {'range': [70, 100], 'color': '#27ae60'}],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': score}
                    }
                ))
                fig.update_layout(height=350, margin=dict(l=30, r=30, t=50, b=30))
                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# ONGLET 2 : Analyse par Lot
# ============================================================
with tab2:
    st.markdown("## 📁 Analyse par lot")
    st.markdown("*Importez un fichier CSV contenant les caractéristiques de plusieurs élèves.*")
    
    # Template
    st.markdown("### 📥 Télécharger un template")
    template_cols = [
        "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
        "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores", "Motivation_Level",
        "Internet_Access", "Tutoring_Sessions", "Family_Income", "Teacher_Quality",
        "School_Type", "Peer_Influence", "Physical_Activity", "Learning_Disabilities",
        "Parental_Education_Level", "Distance_from_Home", "Gender"
    ]
    template_df = pd.DataFrame(columns=template_cols)
    st.download_button(
        "📄 Télécharger le template CSV",
        template_df.to_csv(index=False).encode('utf-8'),
        "template_eleves.csv",
        "text/csv",
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
                    if mode == "🌐 Appeler l'API FastAPI":
                        try:
                            resp = requests.post(API_URL, json=input_row, timeout=5)
                            if resp.status_code == 200:
                                pred = resp.json()["Exam_Score_predicted"]
                            else:
                                pred = np.nan
                        except:
                            pred = np.nan
                    else:
                        if pipeline is None:
                            st.error("Modèle local non chargé.")
                            break
                        pred = pipeline.predict(pd.DataFrame([input_row]))[0]
                    
                    predictions.append(round(pred, 2))
                    progress_bar.progress((idx + 1) / total)
                    status_text.text(f"Traitement : {idx + 1}/{total} élèves...")
                
                batch_df["Score prédit"] = predictions
                batch_df["Niveau de risque"] = batch_df["Score prédit"].apply(
                    lambda x: "🔴 Élevé" if x < 60 else ("🟠 Modéré" if x < 70 else "🟢 Faible")
                )
                
                status_text.text("✅ Prédictions terminées !")
                
                # Résumé
                st.markdown("### 📊 Résumé des résultats")
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    st.metric("Total élèves", len(batch_df))
                with col_s2:
                    st.metric("🔴 Risque élevé", len(batch_df[batch_df["Score prédit"] < 60]))
                with col_s3:
                    st.metric("🟠 Modéré", len(batch_df[
                        (batch_df["Score prédit"] >= 60) & (batch_df["Score prédit"] < 70)
                    ]))
                with col_s4:
                    st.metric("🟢 Bon niveau", len(batch_df[batch_df["Score prédit"] >= 70]))
                
                # Distribution
                fig_dist = px.histogram(
                    batch_df, x="Score prédit", nbins=20,
                    title="Distribution des scores prédits",
                    color_discrete_sequence=['#2a5298'],
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
    
    if pipeline is not None and mode == "💻 Utiliser le modèle local":
        preprocessor = pipeline.named_steps['preprocessor']
        regressor = pipeline.named_steps['regressor']
        feature_names = preprocessor.get_feature_names_out()
        coefs = regressor.coef_
        
        coef_df = pd.DataFrame({'Variable': feature_names, 'Coefficient': coefs})
        coef_df['Abs'] = coef_df['Coefficient'].abs()
        coef_df['Impact'] = coef_df['Coefficient'].apply(lambda x: "✅ Positif" if x > 0 else "❌ Négatif")
        coef_df = coef_df.sort_values('Abs', ascending=False)
        
        # Nettoyage des noms
        coef_df['Variable'] = coef_df['Variable'].str.replace('num__', '').str.replace('ord__', '').str.replace('nom__', '')
        
        col_top1, col_top2 = st.columns([2, 1])
        with col_top1:
            st.markdown("### 🔝 Top 15 des Variables les Plus Influentes")
            fig_coef = px.bar(
                coef_df.head(15).sort_values('Coefficient'),
                x='Coefficient', y='Variable', orientation='h',
                color='Impact',
                color_discrete_map={"✅ Positif": "#27ae60", "❌ Négatif": "#e74c3c"},
                title="Impact des variables sur le score prédit"
            )
            fig_coef.add_vline(x=0, line_width=1, line_color="black")
            fig_coef.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_coef, use_container_width=True)
        
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
        - **Coefficient positif** : augmente le score prédit.
        - **Coefficient négatif** : le diminue.
        - **Valeur absolue** : poids de la variable.
        """)
        
        # Top 5 facteurs de risque / protection
        st.markdown("### ⚠️ Principaux Facteurs de Risque")
        neg = coef_df[coef_df['Coefficient'] < 0].head(5)
        for _, row in neg.iterrows():
            st.markdown(
                f"""
                <div class="resource-card" style="border-left-color: #e74c3c;">
                    <h4>❌ {row['Variable']}</h4>
                    <p>Coefficient : <strong>{row['Coefficient']:.3f}</strong> → 
                    Plus cette variable est élevée, plus le score prédit diminue.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        st.markdown("### 🌟 Principaux Facteurs de Protection")
        pos = coef_df[coef_df['Coefficient'] > 0].head(5)
        for _, row in pos.iterrows():
            st.markdown(
                f"""
                <div class="resource-card" style="border-left-color: #27ae60;">
                    <h4>✅ {row['Variable']}</h4>
                    <p>Coefficient : <strong>{row['Coefficient']:.3f}</strong> → 
                    Plus cette variable est élevée, plus le score prédit augmente.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("⚠️ Passez en mode 'Utiliser le modèle local' pour voir les coefficients.")
    
    # Historique
    if not st.session_state.history.empty:
        st.markdown("---")
        st.markdown("### 📜 Historique des Prédictions")
        st.dataframe(
            st.session_state.history.sort_values("Date", ascending=False),
            use_container_width=True,
        )
        st.download_button(
            "📥 Télécharger l'historique",
            st.session_state.history.to_csv(index=False).encode('utf-8'),
            f"historique_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
        )

# ============================================================
# ONGLET 4 : Ressources & Recommandations
# ============================================================
with tab4:
    st.markdown("## 📚 Ressources Pédagogiques & Recommandations")
    st.markdown("*Des outils pour accompagner chaque élève vers la réussite.*")
    
    # Recommandations personnalisées
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
                "desc": "Planifier des sessions d'étude régulières.",
                "lien": "https://www.pass-ton-bacsn.com/"
            })
        if inp["Access_to_Resources"] == "Low":
            recs.append({
                "titre": "📦 Améliorer l'accès aux ressources",
                "desc": "Orienter vers des bibliothèques et ressources en ligne gratuites.",
                "lien": "https://www.khanacademy.org/"
            })
        if inp["Tutoring_Sessions"] < 2 and score < 70:
            recs.append({
                "titre": "👨‍🏫 Renforcer le tutorat",
                "desc": "Proposer des séances de soutien supplémentaires.",
                "lien": "https://www.superprof.sn/"
            })
        if inp["Learning_Disabilities"] == "Yes":
            recs.append({
                "titre": "🧠 Accompagnement spécialisé",
                "desc": "Consulter un orthophoniste ou psychologue scolaire.",
                "lien": "https://www.tdah-france.fr/"
            })
        if inp["Motivation_Level"] == "Low":
            recs.append({
                "titre": "🔥 Stimuler la motivation",
                "desc": "Fixer des objectifs atteignables, mentorat.",
                "lien": "https://www.reseau-canope.fr/"
            })
        if inp["Internet_Access"] == "No":
            recs.append({
                "titre": "🌐 Faciliter l'accès numérique",
                "desc": "Orienter vers les espaces numériques publics.",
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
                        <a href="{rec['lien']}" target="_blank">🔗 En savoir plus →</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.success("✅ Aucun facteur de risque majeur détecté.")
    
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