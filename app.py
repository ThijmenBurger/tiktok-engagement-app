
import streamlit as st
import pickle
import numpy as np

# Model laden
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# App design
st.title("🎵 TikTok Engagement Voorspeller")
st.subheader("Vul je caption en hashtags in en ontdek hoe goed je post het doet!")

caption = st.text_input("📝 Jouw caption", placeholder="Bijv: Morning routine check ☀️")
hashtags = st.text_input("# Hashtags", placeholder="Bijv: #lifestyle #viral #morning")

if st.button("🔍 Voorspel engagement"):
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
