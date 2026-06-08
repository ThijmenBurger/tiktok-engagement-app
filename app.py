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


def genereer_caption_variaties(beschrijving):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""Je bent een TikTok expert. Schrijf 3 verschillende pakkende Nederlandse TikTok captions op basis van deze videobeschrijving. Elke caption heeft een andere stijl: één grappig, één serieus/inspirerend, één met een vraag.
Geef ook bij elke caption 5 relevante hashtags.

Videobeschrijving: {beschrijving}

Geef je antwoord precies in dit formaat:
CAPTION1: [grappige caption]
HASHTAGS1: [#hashtag1 #hashtag2 #hashtag3 #hashtag4 #hashtag5]
CAPTION2: [serieuze/inspirerende caption]
HASHTAGS2: [#hashtag1 #hashtag2 #hashtag3 #hashtag4 #hashtag5]
CAPTION3: [caption met een vraag]
HASHTAGS3: [#hashtag1 #hashtag2 #hashtag3 #hashtag4 #hashtag5]"""
        }]
    )
    tekst = response.choices[0].message.content
    variaties = []
    current_caption = ""
    current_hashtags = ""
    for regel in tekst.split('\n'):
        regel = regel.strip()
        if regel.startswith("CAPTION1:"):
            current_caption = regel.replace("CAPTION1:", "").strip()
        elif regel.startswith("HASHTAGS1:"):
            current_hashtags = regel.replace("HASHTAGS1:", "").strip()
            variaties.append(("😄 Grappig", current_caption, current_hashtags))
        elif regel.startswith("CAPTION2:"):
            current_caption = regel.replace("CAPTION2:", "").strip()
        elif regel.startswith("HASHTAGS2:"):
            current_hashtags = regel.replace("HASHTAGS2:", "").strip()
            variaties.append(("💪 Inspirerend", current_caption, current_hashtags))
        elif regel.startswith("CAPTION3:"):
            current_caption = regel.replace("CAPTION3:", "").strip()
        elif regel.startswith("HASHTAGS3:"):
            current_hashtags = regel.replace("HASHTAGS3:", "").strip()
            variaties.append(("❓ Vraag", current_caption, current_hashtags))
    return variaties


def analyseer_video(beschrijving, score_label):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""Je bent een TikTok expert. Analyseer deze video en geef advies in het Nederlands.

Videobeschrijving: {beschrijving}
Engagement voorspelling: {score_label}

Geef je antwoord in dit exacte formaat:
WAAROM: [2 zinnen waarom deze video {score_label} engagement krijgt]
STERK: [wat werkt goed aan deze video]
VERBETER: [wat kan beter aan de video zelf]
ALTERNATIEF: [een concreet idee voor een betere video over hetzelfde onderwerp]"""
        }]
    )
    return response.choices[0].message.content


def uitleg_caption(caption, hashtags, score_label):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""Je bent een TikTok expert. Analyseer deze caption en hashtags in het Nederlands.

Caption: {caption}
Hashtags: {hashtags}
Voorspelling: {score_label}

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


def bereken_score(pred, kansen, klassen):
    score_map = {"Low": 0, "Medium": 50, "High": 100}
    base = score_map.get(pred, 50)
    kans_dict = dict(zip(klassen, kansen))
    high_kans = kans_dict.get("High", 0)
    med_kans = kans_dict.get("Medium", 0)
    score = int(high_kans * 100 * 0.6 + med_kans * 100 * 0.3 + base * 0.1)
    return min(max(score, 5), 99)


def kleur_voor_score(score):
    if score >= 70:
        return "#22c55e"
    elif score >= 40:
        return "#f59e0b"
    else:
        return "#ef4444"


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


def toon_score_balk(score):
    kleur = kleur_voor_score(score)
    label = "🔥 Hoog" if score >= 70 else ("👍 Gemiddeld" if score >= 40 else "📉 Laag")
    st.markdown(f"""
    <div style='background:#1e1e2e; border-radius:16px; padding:20px; margin:10px 0;'>
        <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
            <span style='color:white; font-size:16px; font-weight:bold;'>Engagement Score</span>
            <span style='color:{kleur}; font-size:24px; font-weight:bold;'>{score}/100</span>
        </div>
        <div style='background:#2e2e3e; border-radius:999px; height:16px;'>
            <div style='background:{kleur}; width:{score}%; height:16px; border-radius:999px; transition:width 0.5s;'></div>
        </div>
        <div style='text-align:right; margin-top:6px;'>
            <span style='color:{kleur}; font-size:14px;'>{label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def toon_resultaat(pred, kansen, klassen, caption="", hashtags="", toon_uitleg=True):
    score = bereken_score(pred, kansen, klassen)
    toon_score_balk(score)

    with st.expander("📊 Kansen per categorie"):
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


# Styling
st.markdown("""
<style>
    .main { background-color: #0f0f17; }
    h1 { color: #fe2c55 !important; }
    h2, h3 { color: #ffffff !important; }
    .stTabs [data-baseweb="tab"] { color: white; }
    .stTabs [aria-selected="true"] { border-bottom: 3px solid #fe2c55 !important; }
</style>
""", unsafe_allow_html=True)

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
        if st.button("🤖 Genereer captions met AI & voorspel", key="btn2"):
            with st.spinner("AI bedenkt 3 caption variaties..."):
                try:
                    variaties = genereer_caption_variaties(beschrijving)

                    st.subheader("✨ 3 Caption variaties:")
                    beste_score = 0
                    beste_idx = 0

                    for i, (stijl, cap_tekst, hash_tekst) in enumerate(variaties):
                        pred, kansen, klassen = voorspel(cap_tekst, hash_tekst)
                        score = bereken_score(pred, kansen, klassen)
                        if score > beste_score:
                            beste_score = score
                            beste_idx = i

                    for i, (stijl, cap_tekst, hash_tekst) in enumerate(variaties):
                        pred, kansen, klassen = voorspel(cap_tekst, hash_tekst)
                        score = bereken_score(pred, kansen, klassen)
                        kleur = kleur_voor_score(score)
                        winnaar = " 🏆 Beste optie!" if i == beste_idx else ""
                        with st.expander(f"{stijl}{winnaar} — Score: {score}/100"):
                            st.success(f"**Caption:** {cap_tekst}")
                            st.info(f"**Hashtags:** {hash_tekst}")
                            toon_score_balk(score)

                    best_stijl, best_cap, best_hash = variaties[beste_idx]
                    best_pred, best_kansen, best_klassen = voorspel(best_cap, best_hash)

                    st.divider()
                    st.subheader("🎬 Video analyse & aanbevelingen")
                    with st.spinner("AI analyseert je video inhoud..."):
                        analyse = analyseer_video(beschrijving, best_pred)
                        for regel in analyse.split('\n'):
                            if regel.startswith("WAAROM:"):
                                st.info(f"**📊 Waarom deze score:** {regel.replace('WAAROM:', '').strip()}")
                            elif regel.startswith("STERK:"):
                                st.success(f"**💪 Wat werkt goed:** {regel.replace('STERK:', '').strip()}")
                            elif regel.startswith("VERBETER:"):
                                st.warning(f"**🔧 Wat kan beter:** {regel.replace('VERBETER:', '').strip()}")
                            elif regel.startswith("ALTERNATIEF:"):
                                st.info(f"**💡 Probeer dit:** {regel.replace('ALTERNATIEF:', '').strip()}")

                    tijden = get_posting_tijden(beschrijving)
                    with st.expander("⏰ Beste posting tijden"):
                        st.write(f"💡 {tijden['uitleg']}")
                        cols = st.columns(3)
                        for i, tijd in enumerate(tijden['beste']):
                            cols[i].metric("Beste tijd", tijd)

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
            score_a = bereken_score(v_a, k_a, kl_a)
            score_b = bereken_score(v_b, k_b, kl_b)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Resultaat A")
                toon_score_balk(score_a)
                toon_resultaat(v_a, k_a, kl_a, caption_a, hashtags_a, toon_uitleg=False)
            with col2:
                st.subheader("Resultaat B")
                toon_score_balk(score_b)
                toon_resultaat(v_b, k_b, kl_b, caption_b, hashtags_b, toon_uitleg=False)

            st.divider()
            if score_a > score_b:
                st.success(f"🏆 Caption A wint met {score_a} vs {score_b} punten!")
            elif score_b > score_a:
                st.success(f"🏆 Caption B wint met {score_b} vs {score_a} punten!")
            else:
                st.info("🤝 Het is gelijkspel!")
        else:
            st.warning("Vul beide captions in!")
