import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import time
import tempfile

import base64
# --- 1. CONFIGURATION PREMIUM DE LA PAGE ---
st.set_page_config(
    page_title="SKU Vision Pro",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. INJECTION CSS (Le secret du design "Award-Winning") ---
st.markdown("""
    <style>
   
@import url('https://fonts.googleapis.com/css2?family=Bagel+Fat+One&family=Borel&family=Bricolage+Grotesque:opsz,wght@12..96,200..800&family=Bruno+Ace&family=Genos:ital,wght@0,100..900;1,100..900&family=Kavoon&family=Lilita+One&family=Oi&family=Righteous&family=Rubik+Dirt&family=Rubik+Vinyl&display=swap');
.bruno-ace-regular {
  font-family: "Bruno Ace", sans-serif;
  font-weight: 400;
  font-style: normal;
}
            /* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #000000;
}

/* Texte dans la sidebar */
[data-testid="stSidebar"] * {
    color: white;
}

/* Slider label */
[data-testid="stSidebar"] .stSlider label {
    color: white;
}

/* Caption */
[data-testid="stSidebar"] .stCaption {
    color: #cccccc;
}


    .stApp {
        background-color: #FFFFFF;
             
        color: #000000;
        font-family:  "Bruno Ace", sans-serif;
    }
        
    /* Animation d'entrée fluide */
    @keyframes fadeUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .main .block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem;
    animation: fadeUp 8s cubic-bezier(0.16, 1, 0.3, 1);
}
            /* Supprimer l’espace en haut */
section[data-testid="stMain"] > div {
    padding-top: 0rem !important;
}

/* Alternative (sécurité) */
div.block-container {
    padding-top: 0rem !important;
}

    /* Typographie élégante */
    h1, h2, h3 {
        font-weight: 900 !important;
        letter-spacing: 1px;
        
    }
    h1 {
        font-size: 3.5rem !important;
        border-bottom: 1px solid #333;
        padding-bottom: 20px;
              font-family: "Bruno Ace", sans-serif !important;
        margin-bottom: 40px;
                margin-top: 0 !important;
    padding-top: 0 !important;

        font-style: uppercase;
    }
    h2,h3,h4,h5 {
    font-family: "Genos", sans-serif !important;}
    

    /* Cacher les éléments superflus de Streamlit pour un look "App native" */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
      button[kind="header"] svg {
    display: none;
}


    /* Cartes de métriques stylisées (Le compteur de produits) */
    div[data-testid="metric-container"] {
        background-color: #1A1A1A;
        border: 1px solid #333;
        padding: 5% 5% 5% 10%;
        
        border-radius: 8px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        transition: transform 0.3s ease;
        color: #FFFFFF;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-color: #777;
    }
    
    /* Uploader customisé */
    [data-testid="stFileUploader"] button {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #ffffff !important;
        border-color: #000000 !important;
        color: #000000 !important;
        transform: scale(1.02);
    }
    
    /* Couleurs des métriques */
    [data-testid="stMetricValue"] { color: #666 !important;  font-family:"Bruno Ace", sans-serif !important; }
            [data-testid="stMetricValue"] * {
    font-family: 'Bruno Ace', sans-serif !important;
}
    [data-testid="stMetricLabel"] { color: #888888 !important; }
    [data-testid="stMetricDelta"] { color: #cccccc !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CHARGEMENT DU MODÈLE (En cache pour la vitesse) ---
@st.cache_resource

def load_model():
    # Ton modèle de 50 époques
    model_path = './best.pt' 
    return YOLO(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Erreur d'initialisation du moteur IA : {e}")

# --- 4. SIDEBAR (Contrôles) ---
with st.sidebar:
    st.markdown("###  MOTEUR D'INFÉRENCE")
    conf_threshold = st.slider("Sensibilité IA (Confidence)", 0.05, 1.00, 0.25, 0.05)
    st.markdown("---")
    
    # NOUVEAU : Contrôle métier pour la rupture de stock
    st.markdown("###  PARAMÈTRES MÉTIER")
    density_threshold = st.slider("Seuil d'Alerte Rupture (%)", 10, 100, 50, 5)
    st.markdown("---")
    
    st.caption("Propulsé par YOLOv8n | Optimisé pour SKU-110K")
    #st.caption("Développé par Ilyass EL-OGRI")

# --- 5. INTERFACE PRINCIPALE ---
st.title("SKU Vision Pro.")
st.markdown("### Analyse topologique de rayonnage en temps réel.")

uploaded_file = st.file_uploader("CHARGEZ UNE CAPTURE VISUELLE (JPG, PNG, MP4)", type=['jpg', 'jpeg', 'png','webp','mp4'])
fond_ecran = st.empty()
if uploaded_file is not None:
    fond_ecran.empty()
    
   
    #  CAS 1 : C'EST UNE VIDÉO (.mp4)

    if uploaded_file.name.endswith('.mp4'):
        
        # Création des mêmes colonnes que pour l'image
        col1, spacer, col2 = st.columns([1, 0.1, 1])
        
        with col1:
            st.markdown("<h4 style='text-align: center; color: #888;'>FLUX VIDÉO SOURCE</h4>", unsafe_allow_html=True)
            ecran_radar = st.empty() # Boîte vide pour la vidéo qui tourne
            
        with col2:
            st.markdown("<h4 style='text-align: center; color: #888;'>ANALYSE IA & BUSINESS</h4>", unsafe_allow_html=True)
            panneau_metriques = st.empty() # Boîte vide pour les métriques qui s'actualisent
            
        # Traitement de la vidéo
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 1. PRÉDICTION SUR LA FRAME
            results = model.predict(source=frame, conf=conf_threshold, verbose=False)
            
            # 2. DESSIN DE LA FRAME
            res_plotted = results[0].plot(labels=False, conf=False, line_width=3)
            res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            ecran_radar.image(res_plotted_rgb, use_container_width=True)
            
            # 3. CALCULS INTELLIGENTS (Ton code exact)
            nb_produits = len(results[0].boxes)
            img_height, img_width = results[0].orig_shape
            surface_totale = img_width * img_height
            
            surface_produits = 0
            if nb_produits > 0:
                boxes_data = results[0].boxes.xywh.cpu().numpy()
                for box in boxes_data:
                    w, h = box[2], box[3]
                    surface_produits += (w * h)
                    
            densite_brute = (surface_produits / surface_totale) * 100
            taux_metier = (densite_brute / 70) * 100
            taux_affichage = min(taux_metier, 100.0)
            
            # 4. AFFICHAGE DES MÉTRIQUES EN TEMPS RÉEL (Dans la boîte vide)
            with panneau_metriques.container():
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric(label="Unités Détectées", value=nb_produits)
                with m_col2:
                    st.metric(label="Taux de Remplissage", value=f"{taux_affichage:.1f} %")
                
                st.progress(int(taux_affichage) / 100.0)
                
                if taux_affichage < density_threshold:
                    st.error(f" ALERTE RUPTURE : Le rayon est rempli à seulement {taux_affichage:.1f}%. (Seuil critique : {density_threshold}%)")
                else:
                    st.success(f" CONFORME : Le rayonnage est considéré comme plein à {taux_affichage:.1f}%.")
                    
        cap.release()
        st.success("Analyse du flux vidéo terminée.")


    #  CAS 2 : C'EST UNE IMAGE (.jpg, .png, etc.)
   
    else:
        image = Image.open(uploaded_file)
        
        col1, spacer, col2 = st.columns([1, 0.1, 1])
        
        with col1:
            st.markdown("<h4 style='text-align: center; color: #888;'>IMAGE SOURCE</h4>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("<h4 style='text-align: center; color: #888;'>ANALYSE IA & BUSINESS</h4>", unsafe_allow_html=True)
         
            with st.spinner('Extraction des caractéristiques spatiales...'):
                time.sleep(0.5) 
                
                results = model.predict(source=image, conf=conf_threshold)
                
                res_plotted = results[0].plot(labels=False, conf=False, line_width=3)
                res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                st.image(res_plotted_rgb, use_container_width=True)
                
                nb_produits = len(results[0].boxes)
                
                img_height, img_width = results[0].orig_shape
                surface_totale = img_width * img_height
                
                surface_produits = 0
                if nb_produits > 0:
                    boxes_data = results[0].boxes.xywh.cpu().numpy()
                    for box in boxes_data:
                        w, h = box[2], box[3]
                        surface_produits += (w * h)
                
                densite_brute = (surface_produits / surface_totale) * 100
                taux_metier = (densite_brute / 70) * 100
                taux_affichage = min(taux_metier, 100.0) 
                
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric(label="Unités Détectées", value=nb_produits)
                with m_col2:
                    st.metric(label="Taux de Remplissage", value=f"{taux_affichage:.1f} %")
                
                st.progress(int(taux_affichage) / 100.0)
                
                if taux_affichage < density_threshold:
                    st.error(f" ALERTE RUPTURE : Le rayon est rempli à seulement {taux_affichage:.1f}%. (Seuil critique : {density_threshold}%)")
                else:
                    st.success(f" CONFORME : Le rayonnage est considéré comme plein à {taux_affichage:.1f}%.")
                    
        st.markdown("---")
        st.caption("Inférence et calcul géométrique terminés en " + str(round(results[0].speed['inference'], 1)) + " ms.")

else:
    # État vide élégant
    st.markdown("<br><br><h5 style='text-align: center; color: #444;'>En attente d'image ou de vidéo...</h5>", unsafe_allow_html=True)
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    try:
        # On lit ton fichier local
        img_base64 = get_base64_of_bin_file('bg1.png')
        # On crée le lien lisible par le navigateur web
        url_image = f"data:image/png;base64,{img_base64}"
        with fond_ecran.container():
        # 2. INJECTION HTML/CSS CORRIGÉE
            st.markdown(f"""
            <style>
            .footer-image-container {{
                position: fixed;
                bottom: 0px;
                right: 0px;
                width: 80vw;
                height: 300px;
                z-index: 999;
                pointer-events: none;
            }}
            .footer-image-container img {{
                width: 100%;
                height: 300px;
                display: block;
                opacity: 1;
             
                transform: scaleX(2);
                -webkit-mask-image: linear-gradient(to top, rgba(0,0,0,1) 50%, rgba(0,0,0,0) 100%);
                mask-image: linear-gradient(to top, rgba(0,0,0,1) 50%, rgba(0,0,0,0) 100%);
            
               
                
                
            }}
            </style>
            
            <div class="footer-image-container">
                <img src="{url_image}" alt="Image de fond">
            </div>
        """, unsafe_allow_html=True)
        
    except FileNotFoundError:
        st.error(" Fichier bg.png introuvable. Vérifie qu'il est dans le même dossier que le script.")