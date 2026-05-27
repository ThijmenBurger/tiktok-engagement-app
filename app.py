import streamlit as st
import pickle
import numpy as np
import cv2
import tempfile
import os
import random

# Model laden
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Caption generator op basis van beschrijving
def genereer_caption_en_hashtags(beschrijving):
    beschrijving = beschrijving.lower()
    
    # Hashtag bibliotheek
    hashtag_sets = {
        'sport': ['#sport', '#fitness', '#training', '#workout', '#athlete', '#fyp', '#viral'],
        'voetbal': ['#voetbal', '#football', '#soccer', '#goals', '#fyp', '#viral', '#sport'],
        'gym': ['#gym', '#fitness', '#gains', '#workout', '#bodybuilding', '#fyp', '#viral'],
        'food': ['#food', '#foodie', '#lekker', '#cooking', '#foodtok', '#fyp', '#viral'],
        'lifestyle': ['#lifestyle', '#daily', '#vlog', '#dayinmylife', '#fyp', '#viral'],
        'gaming': ['#gaming', '#gamer', '#game', '#ps5', '#xbox', '#fyp', '#viral'],
        'muziek': ['#muziek', '#music', '#dance', '#fyp', '#viral', '#trending'],
        'grappig': ['#funny', '#humor', '#lol', '#grappig', '#fyp', '#viral'],
    }
    
    # Caption templates
    caption_templates = {
        'sport': [
            "Vandaag alles gegeven op het veld 💪",
            "Geen excuses, alleen resultaten 🏆",
            "Training never lies 🔥",
        ],
        'voetbal': [
            "Het mooiste doelpunt van mijn leven ⚽🔥",
            "Zo hoort voetbal gespeeld te worden! ⚽",
            "Skills on point vandaag ⚽💫",
        ],
        'gym': [
            "Geen pijn, geen winst 💪🔥",
            "Vandaag weer een PR gebroken 🏋️",
            "De grind stopt nooit 💪",
        ],
        'food': [
            "Dit moet je echt een keer proberen 😍🍕",
            "Zelfgemaakt en zo lekker 🍽️✨",
            "Foodie heaven right here 😋",
        ],
        'lifestyle': [
            "Een dag uit mijn leven ✨",
            "Dit is hoe ik mijn dag begin 🌅",
            "Life is good als je dit hebt 😊",
        ],
        'gaming': [
            "Deze clip is te gek 🎮🔥",
            "Niemand kan me stoppen vandaag 🎮💪",
            "Pro moves only 🎮✨",
        ],
        'muziek': [
            "Dit nummer raakt elke keer weer 🎵❤️",
            "Kan niet stoppen met dansen 🎶🔥",
            "Vibes alleen maar vibes 🎵✨",
        ],
        'grappig': [
            "Ik kan er zelf ook niet om ophouden met lachen 😂",
            "Dit had ik niet zien aankomen 😂🔥",
            "Waarom overkomt mij dit altijd 😂",
        ],
    }
    
    # Detecteer onderwerp
    gekozen_categorie = 'lifestyle'
    for categorie in hashtag_sets:
        if categorie in beschrijving:
            gekozen_categorie = categorie
            break
    
    caption = random.choice(caption_templates[gekozen_categorie])
    hashtags = ' '.join(random.sample(hashtag_sets[gekozen_categorie], min(5, len(hashtag_sets[gekozen_categorie]))))
    
    return caption, hashtags

# App design
st.title("🎵 TikTok Engagement Voorspeller")
st.subheader("Upload een video of vul je caption in en ontdek hoe goed je post het doet!")

# Tabs
tab1, tab2 = st.tabs(["📝 Caption invoeren", "🎬 Video uploaden"])

with tab1:
    caption = st.text_input("📝 Jouw caption", placeholder="Bijv: Morning routine check ☀️")
    hashtags = st.text_input("# Hashtags", placeholder="Bijv: #lifestyle #viral #morning")

    if st.button("🔍 Voorspel engagement", key="btn1"):
        if caption == "" and hashtags == "":
            st.warning("Vul eerst een caption of hashtags in!")
        else:
            tekst = caption + " " + hashtags
            vector = vectorizer.transform([tekst])
            voorspelling = model.predict(vector)[0]
            kansen = model.predict_proba(vector)[0]
            klassen = model.classes_

            if voorspelling == "High":
                st.success("🔥 Hoge engagement verwacht!")
            elif voorspelling == "Medium":
                st.info("👍 Gemiddelde engagement verwacht!")
            else:
                st.warning("📉 Lage engagement verwacht")

            st.subheader("Kansen per categorie:")
            for klasse, kans in zip(klassen, kansen):
                st.progress(float(kans), text=f"{klasse}: {kans:.0%}")

            st.subheader("💡 Tips:")
            if voorspelling != "High":
                st.write("- Voeg meer trending hashtags toe zoals #viral of #fyp")
                st.write("- Maak je caption korter en pakkender")
                st.write("- Gebruik een emoji aan het begin")

with tab2:
    st.write("Upload je TikTok video en beschrijf kort wat erin gebeurt.")
    
    video_file = st.file_uploader("🎬 Upload je video", type=['mp4', 'mov', 'avi'])
    beschrijving = st.text_input("📝 Beschrijf je video kort", placeholder="Bijv: voetbal doelpunt, gym workout, grappig moment")
    
    if video_file and beschrijving:
        # Video opslaan en frame extraheren
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name
        
        # Frame uit video halen
        cap = cv2.VideoCapture(tmp_path)
        ret, frame = cap.read()
        cap.release()
        os.unlink(tmp_path)
        
        if ret:
            # Frame tonen
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="📸 Screenshot uit je video", use_container_width=True)
        
        if st.button("🤖 Genereer caption & voorspel", key="btn2"):
            # Caption genereren
            gegenereerde_caption, gegenereerde_hashtags = genereer_caption_en_hashtags(beschrijving)
            
            st.subheader("✨ Gegenereerde caption:")
            st.info(f"**Caption:** {gegenereerde_caption}")
            st.info(f"**Hashtags:** {gegenereerde_hashtags}")
            
            # Engagement voorspellen
            tekst = gegenereerde_caption + " " + gegenereerde_hashtags
            vector = vectorizer.transform([tekst])
            voorspelling = model.predict(vector)[0]
            kansen = model.predict_proba(vector)[0]
            klassen = model.classes_
            
            st.subheader("🎯 Engagement voorspelling:")
            if voorspelling == "High":
                st.success("🔥 Hoge engagement verwacht!")
            elif voorspelling == "Medium":
                st.info("👍 Gemiddelde engagement verwacht!")
            else:
                st.warning("📉 Lage engagement verwacht")
            
            for klasse, kans in zip(klassen, kansen):
                st.progress(float(kans), text=f"{klasse}: {kans:.0%}")
