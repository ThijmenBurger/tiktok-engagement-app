import streamlit as st
import pickle
import cv2
import tempfile
import os
from groq import Groq

# Groq instellen
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ML Model laden
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def genereer_caption_met_groq(beschrijving):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""Je bent een TikTok expert. Schrijf op basis van deze videobeschrijving:
1. Een pakkende Nederlandse TikTok caption (max 1 zin, met emoji's)
2. 5 relevante hashtags

Videobeschrijving: {beschrijving}

Geef je antwoord precies in dit formaat:
CAPTION: [jouw caption hier]
HASHTAGS: [#hashtag1 #hashtag2 #hashtag3 #hashtag4 #hashtag5]"""
        }]
    )
    tekst = response.choices[0].message.content
    caption = ""
    hashtags = ""
    for regel in tekst.split('\n'):
        if regel.startswith("CAPTION:"):
            caption = regel.replace("CAPTION:", "").strip()
        elif regel.startswith("HASHTAGS:"):
            hashtags = regel.replace("HASHTAGS:", "").strip()
    return caption, hashtags

def voorspel(caption, hashtags):
    tekst = caption + " " + hashtags
    vector = vectorizer.transform([tekst])
    voorspelling = model.predict(vector)[0]
    kansen = model.predict_proba(vector)[0]
    klassen = model.classes_
    return voorspelling, kansen, klassen

def toon_resultaat(voorspelling, kansen, klassen):
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
    if voorspelling == "High":
        st.write("✅ Je caption ziet er goed uit! Post hem op een piekmoment.")
        st.write("✅ Reageer snel op de eerste comments voor extra boost.")
    elif voorspelling == "Medium":
        st.write("- Voeg #fyp of #viral toe aan je hashtags")
        st.write("- Maak je caption iets korter en pakkender")
        st.write("- Gebruik een emoji aan het begin")
    else:
        st.write("- Herformuleer je caption — maak het persoonlijker")
        st.write("- Gebruik trending hashtags zoals #fyp #viral #foryou")
        st.write("- Stel een vraag in je caption voor meer reacties")

st.title("🎵 TikTok Engagement Voorspeller")
st.subheader("Upload een video of vul je caption in en ontdek hoe goed je post het doet!")

tab1, tab2 = st.tabs(["📝 Caption invoeren", "🎬 Video uploaden"])

with tab1:
    caption = st.text_input("📝 Jouw caption", placeholder="Bijv: Morning routine check ☀️")
    hashtags = st.text_input("# Hashtags", placeholder="Bijv: #lifestyle #viral #morning")

    if st.button("🔍 Voorspel engagement", key="btn1"):
        if caption == "" and hashtags == "":
            st.warning("Vul eerst een caption of hashtags in!")
        else:
            voorspelling, kansen, klassen = voorspel(caption, hashtags)
            toon_resultaat(voorspelling, kansen, klassen)

with tab2:
    st.write("Upload je TikTok video en beschrijf kort wat erin gebeurt.")
    video_file = st.file_uploader("🎬 Upload je video", type=['mp4', 'mov', 'avi'])
    beschrijving = st.text_input("📝 Beschrijf je video", placeholder="Bijv: Denzel Dumfries transfer naar Real Madrid, grappige video van slecht naar goed")

    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        ret, frame = cap.read()
        cap.release()
        os.unlink(tmp_path)
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="📸 Screenshot uit je video", width=400)

    if video_file and beschrijving:
        if st.button("🤖 Genereer caption met AI & voorspel", key="btn2"):
            with st.spinner("AI bedenkt een caption..."):
                try:
                    gegenereerde_caption, gegenereerde_hashtags = genereer_caption_met_groq(beschrijving)
                    st.subheader("✨ Door AI gegenereerde caption:")
                    st.success(f"**Caption:** {gegenereerde_caption}")
                    st.info(f"**Hashtags:** {gegenereerde_hashtags}")
                    voorspelling, kansen, klassen = voorspel(gegenereerde_caption, gegenereerde_hashtags)
                    st.subheader("🎯 Engagement voorspelling:")
                    toon_resultaat(voorspelling, kansen, klassen)
                except Exception as e:
                    st.error(f"Er ging iets mis: {e}")
