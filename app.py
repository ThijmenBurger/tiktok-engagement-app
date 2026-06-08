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


def analyseer_video(beschrijving, voorspelling):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""Je bent een TikTok expert. Analyseer deze video en geef advies in het Nederlands.

Videobeschrijving: {beschrijving}
Engagement voorspelling: {voorspelling}

Geef je antwoord in dit exacte formaat:
WAAROM: [2 zinnen waarom deze video {voorspelling} engagement krijgt]
STERK: [wat werkt goed aan deze video]
VERBETER: [wat kan beter aan de video zelf]
ALTERNATIEF: [een concreet idee voor een betere video over hetzelfde onderwerp]"""
        }]
    )
    return response.choices[0].message.content


def uitleg_caption(caption, hashtags, voorspelling):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""Je bent een TikTok expert. Analyseer deze caption en hashtags in het Nederlands.

Caption: {caption}
Hashtags: {hashtags}
Voorspelling: {voorspelling}

Formaat:
PUNT1: [uitleg punt 1]
PUNT2: [uitleg punt 2]
PUNT3: [uitleg punt 3]"""
        }]
    )
    return response.choices[0].message.content


def voorspel(caption, hashtags):
    tekst = caption + " " + hashtags
    vector = vectorizer.transform([tekst])
    pred = model.predict(vector)[0]
    kansen = model.predict_proba(vector)[0]
    klassen = model.classes_
    return pred, kansen, klassen


POSTING_TIJDEN = {
    "sport": {"beste": ["18:00", "20:00", "21:00"], "uitleg": "Na het werk/school kijken mensen sportcontent"},
    "voetbal": {"beste": ["18:00", "20:00", "22:00"], "uitleg": "Avonden en na wedstrijden zijn piektijden"},
    "gym": {"beste": ["06:00", "12:00", "18:00"], "uitleg": "Voor/na het sporten zijn mensen actief op TikTok"},
    "food": {"beste": ["11:30", "17:30", "20:00"], "uitleg": "Rond etenstijden scoort foodcontent het best"},
    "lifestyle": {"beste": ["07:00", "12:00", "21:00"], "uitleg": "Ochtend, lunch en avond zijn piekmomenten"},
    "gaming": {"beste": ["16:00", "20:00", "23:00"], "uitleg": "Middag en late avond zijn gamers actief"},
    "muziek": {"beste": ["15:00", "19:00", "22:00"], "uitleg": "Namiddag en avond voor muziekcontent"},
    "algemeen": {"beste": ["07:00", "12:00", "19:00"], "uitleg": "Ochtend, lunch en avond zijn algemene piektijden"},
}


def get_posting_tijden(tekst):
    tekst = tekst.lower()
    for categorie in POSTING_TIJDEN:
        if categorie in tekst:
            return POSTING_TIJDEN[categorie]
    return POSTING_TIJDEN["algemeen"]


def toon_resultaat(pred, kansen, klassen, caption="", hashtags="", toon_uitleg=True):
    if pred == "High":
        st.success("🔥 Hoge engagement verwacht!")
    elif pred == "Medium":
        st.info("👍 Gemiddelde engagement verwacht!")
    else:
        st.warning("📉 Lage engagement verwacht")

    st.subheader("Kansen per categorie:")
    for klasse, kans in zip(klassen, kansen):
        st.progress(float(kans), text=f"{klasse}: {kans:.0%}")

    if toon_uitleg and caption:
        with st.expander("🔍 Waarom scoort deze caption zo?"):
            with st.spinner("AI analyseert je caption..."):
                uitleg = uitleg_caption(caption, hashtags, pred)
                for regel in uitleg.split('\n'):
                    if regel.startswith("PUNT"):
                        inhoud = regel.split(":", 1)[-1].strip()
                        st.write(f"• {inhoud}")

    tijden = get_posting_tijden(caption + " " + hashtags)
    with st.expander("⏰ Beste posting tijden"):
        st.write(f"💡 {tijden['uitleg']}")
        cols = st.columns(3)
        for i, tijd in enumerate(tijden['beste']):
            cols[i].metric("Beste tijd", tijd)

    st.subheader("💡 Tips:")
    if pred == "High":
        st.write("✅ Je caption ziet er goed uit! Post hem op een piekmoment.")
        st.write("✅ Reageer snel op de eerste comments voor extra boost.")
    elif pred == "Medium":
        st.write("- Voeg #fyp of #viral toe aan je hashtags")
        st.write("- Maak je caption iets korter en pakkender")
        st.write("- Gebruik een emoji aan het begin")
    else:
        st.write("- Herformuleer je caption — maak het persoonlijker")
        st.write("- Gebruik trending hashtags zoals #fyp #viral #foryou")
        st.write("- Stel een vraag in je caption voor meer reacties")


# App design
st.title("🎵 TikTok Engagement Voorspeller")
st.subheader("Upload een video of vul je caption in en ontdek hoe goed je post het doet!")

tab1, tab2, tab3 = st.tabs(["📝 Caption invoeren", "🎬 Video uploaden", "⚔️ Caption vergelijker"])

with tab1:
    caption = st.text_input("📝 Jouw caption", placeholder="Bijv: Morning routine check ☀️")
    hashtags = st.text_input("# Hashtags", placeholder="Bijv: #lifestyle #viral #morning")

    if st.button("🔍 Voorspel engagement", key="btn1"):
        if caption == "" and hashtags == "":
            st.warning("Vul eerst een caption of hashtags in!")
        else:
            pred, kansen, klassen = voorspel(caption, hashtags)
            toon_resultaat(pred, kansen, klassen, caption, hashtags)

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
            with st.spinner("AI analyseert je video..."):
                try:
                    gegenereerde_caption, gegenereerde_hashtags = genereer_caption_met_groq(beschrijving)

                    st.subheader("✨ Door AI gegenereerde caption:")
                    st.success(f"**Caption:** {gegenereerde_caption}")
                    st.info(f"**Hashtags:** {gegenereerde_hashtags}")

                    pred, kansen, klassen = voorspel(gegenereerde_caption, gegenereerde_hashtags)

                    st.subheader("🎯 Engagement voorspelling:")
                    toon_resultaat(pred, kansen, klassen, gegenereerde_caption, gegenereerde_hashtags)

                    st.divider()
                    st.subheader("🎬 Video analyse & aanbevelingen")
                    with st.spinner("AI analyseert je video inhoud..."):
                        analyse = analyseer_video(beschrijving, pred)
                        for regel in analyse.split('\n'):
                            if regel.startswith("WAAROM:"):
                                st.info(f"**📊 Waarom deze score:** {regel.replace('WAAROM:', '').strip()}")
                            elif regel.startswith("STERK:"):
                                st.success(f"**💪 Wat werkt goed:** {regel.replace('STERK:', '').strip()}")
                            elif regel.startswith("VERBETER:"):
                                st.warning(f"**🔧 Wat kan beter:** {regel.replace('VERBETER:', '').strip()}")
                            elif regel.startswith("ALTERNATIEF:"):
                                st.info(f"**💡 Probeer dit:** {regel.replace('ALTERNATIEF:', '').strip()}")

                except Exception as e:
                    st.error(f"Er ging iets mis: {e}")

with tab3:
    st.write("Test twee captions tegen elkaar en kijk welke beter scoort!")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Caption A")
        caption_a = st.text_input("📝 Caption A", placeholder="Bijv: Deze goal is van een andere wereld ⚽🔥", key="ca")
        hashtags_a = st.text_input("# Hashtags A", placeholder="Bijv: #voetbal #viral #fyp", key="ha")

    with col2:
        st.subheader("Caption B")
        caption_b = st.text_input("📝 Caption B", placeholder="Bijv: POV: je scoort in de laatste minuut 😱", key="cb")
        hashtags_b = st.text_input("# Hashtags B", placeholder="Bijv: #football #goals #trending", key="hb")

    if st.button("⚔️ Vergelijk captions", key="btn3"):
        if caption_a and caption_b:
            v_a, k_a, kl_a = voorspel(caption_a, hashtags_a)
            v_b, k_b, kl_b = voorspel(caption_b, hashtags_b)

            score_map = {"High": 3, "Medium": 2, "Low": 1}
            score_a = score_map[v_a]
            score_b = score_map[v_b]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Resultaat A")
                toon_resultaat(v_a, k_a, kl_a, caption_a, hashtags_a, toon_uitleg=False)
            with col2:
                st.subheader("Resultaat B")
                toon_resultaat(v_b, k_b, kl_b, caption_b, hashtags_b, toon_uitleg=False)

            st.divider()
            if score_a > score_b:
                st.success("🏆 Caption A wint!")
            elif score_b > score_a:
                st.success("🏆 Caption B wint!")
            else:
                st.info("🤝 Het is gelijkspel!")
        else:
            st.warning("Vul beide captions in!")
