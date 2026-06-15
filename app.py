import streamlit as st
import pickle
import cv2
import tempfile
import os
from groq import Groq

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


def genereer_caption_variaties(beschrijving, taal="Nederlands"):
    taal_instructie = "Nederlandse" if taal == "Nederlands" else "English"
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""You are a TikTok expert. Write 3 different {taal_instructie} TikTok captions based on this video description. One funny, one serious/inspiring, one with a question. Include 5 relevant hashtags per caption.

Video description: {beschrijving}

Exact format:
CAPTION1: [funny caption]
HASHTAGS1: [#tag1 #tag2 #tag3 #tag4 #tag5]
CAPTION2: [serious/inspiring caption]
HASHTAGS2: [#tag1 #tag2 #tag3 #tag4 #tag5]
CAPTION3: [question caption]
HASHTAGS3: [#tag1 #tag2 #tag3 #tag4 #tag5]"""
        }]
    )
    tekst = response.choices[0].message.content
    variaties = []
    current_caption = ""
    for regel in tekst.split('\n'):
        regel = regel.strip()
        if regel.startswith("CAPTION1:"):
            current_caption = regel.replace("CAPTION1:", "").strip()
        elif regel.startswith("HASHTAGS1:"):
            variaties.append(("😄 Grappig" if taal == "Nederlands" else "😄 Funny", current_caption, regel.replace("HASHTAGS1:", "").strip()))
        elif regel.startswith("CAPTION2:"):
            current_caption = regel.replace("CAPTION2:", "").strip()
        elif regel.startswith("HASHTAGS2:"):
            variaties.append(("💪 Inspirerend" if taal == "Nederlands" else "💪 Inspiring", current_caption, regel.replace("HASHTAGS2:", "").strip()))
        elif regel.startswith("CAPTION3:"):
            current_caption = regel.replace("CAPTION3:", "").strip()
        elif regel.startswith("HASHTAGS3:"):
            variaties.append(("❓ Vraag" if taal == "Nederlands" else "❓ Question", current_caption, regel.replace("HASHTAGS3:", "").strip()))
    return variaties


def analyseer_video(beschrijving, score_label, taal="Nederlands"):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""You are a TikTok expert. Analyze this video in {'Dutch' if taal == 'Nederlands' else 'English'}.

Video description: {beschrijving}
Engagement prediction: {score_label}

Exact format:
WAAROM: [why this video gets {score_label} engagement]
STERK: [what works well]
VERBETER: [what can be improved]
ALTERNATIEF: [concrete idea for a better video on same topic]"""
        }]
    )
    return response.choices[0].message.content


def uitleg_caption(caption, hashtags, score_label, taal="Nederlands"):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""You are a TikTok expert. Analyze this caption in {'Dutch' if taal == 'Nederlands' else 'English'}.

Caption: {caption}
Hashtags: {hashtags}
Prediction: {score_label}

Format:
PUNT1: [explanation]
PUNT2: [explanation]
PUNT3: [explanation]"""
        }]
    )
    return response.choices[0].message.content


def get_trends(categorie, taal="Nederlands"):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""You are a TikTok trend expert. Give 5 current trending topics/formats for the {categorie} niche on TikTok in {'Dutch' if taal == 'Nederlands' else 'English'}.

Format:
TREND1: [trending topic/format with short explanation]
TREND2: [trending topic/format with short explanation]
TREND3: [trending topic/format with short explanation]
TREND4: [trending topic/format with short explanation]
TREND5: [trending topic/format with short explanation]"""
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
    kans_dict = dict(zip(klassen, kansen))
    high_kans = kans_dict.get("High", 0)
    med_kans = kans_dict.get("Medium", 0)
    low_kans = kans_dict.get("Low", 0)
    score = int(high_kans * 95 + med_kans * 55 + low_kans * 10)
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
    "football": {"beste": ["18:00", "20:00", "22:00"], "uitleg": "Evenings and after matches are peak times"},
    "gym": {"beste": ["06:00", "12:00", "18:00"], "uitleg": "Voor/na het sporten zijn mensen actief op TikTok"},
    "food": {"beste": ["11:30", "17:30", "20:00"], "uitleg": "Rond etenstijden scoort foodcontent het best"},
    "lifestyle": {"beste": ["07:00", "12:00", "21:00"], "uitleg": "Ochtend, lunch en avond zijn piekmomenten"},
    "gaming": {"beste": ["16:00", "20:00", "23:00"], "uitleg": "Middag en late avond zijn gamers actief"},
    "muziek": {"beste": ["15:00", "19:00", "22:00"], "uitleg": "Namiddag en avond voor muziekcontent"},
    "music": {"beste": ["15:00", "19:00", "22:00"], "uitleg": "Afternoon and evening for music content"},
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
            <div style='background:{kleur}; width:{score}%; height:16px; border-radius:999px;'></div>
        </div>
        <div style='text-align:right; margin-top:6px;'>
            <span style='color:{kleur}; font-size:14px;'>{label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def toon_resultaat(pred, kansen, klassen, caption="", hashtags="", toon_uitleg=True, taal="Nederlands"):
    score = bereken_score(pred, kansen, klassen)
    toon_score_balk(score)

    with st.expander("📊 Kansen per categorie"):
        for klasse, kans in zip(klassen, kansen):
            st.progress(float(kans), text=f"{klasse}: {kans:.0%}")

    if toon_uitleg and caption:
        with st.expander("🔍 Waarom scoort deze caption zo?"):
            with st.spinner("AI analyseert..."):
                uitleg = uitleg_caption(caption, hashtags, pred, taal)
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

    if score >= 70:
        st.write("✅ Je caption ziet er goed uit! Post hem op een piekmoment.")
        st.write("✅ Reageer snel op de eerste comments voor extra boost.")
    elif score >= 40:
        st.write("- Voeg #fyp of #viral toe aan je hashtags")
        st.write("- Maak je caption iets korter en pakkender")
        st.write("- Gebruik een emoji aan het begin")
    else:
        st.write("- Herformuleer je caption — maak het persoonlijker")
        st.write("- Gebruik trending hashtags zoals #fyp #viral #foryou")
        st.write("- Stel een vraag in je caption voor meer reacties")


# Styling + Logo
st.markdown("""
<style>
    .main { background-color: #0f0f17; }
    .stTabs [data-baseweb="tab"] { color: white; }
    .stTabs [aria-selected="true"] { border-bottom: 3px solid #fe2c55 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='display:flex; align-items:center; gap:20px; padding:10px 0 20px 0;'>
    <svg width="70" height="70" viewBox="0 0 200 200">
        <rect width="200" height="200" rx="40" fill="#fe2c55"/>
        <polygon points="65,55 65,145 150,100" fill="white"/>
        <circle cx="155" cy="55" r="42" fill="#25f4ee"/>
        <circle cx="155" cy="55" r="32" fill="#0f0f17"/>
        <polyline points="140,55 152,68 172,40" fill="none" stroke="#25f4ee" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <div>
        <div style='font-size:32px; font-weight:900; line-height:1;'><span style="color:#25f4ee;">Viral</span><span style="color:#fe2c55;">Check</span> <span style="color:white;">AI</span></div>
        <div style='font-size:13px; color:#888; letter-spacing:3px; margin-top:4px;'>ENGAGEMENT PREDICTOR</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Taal + Profiel sidebar
with st.sidebar:
    st.markdown("### ⚙️ Instellingen")
    taal = st.selectbox("🌍 Taal / Language", ["Nederlands", "English"])
    st.markdown("---")
    st.markdown("### 👤 Jouw profiel")
    profiel = st.selectbox("Wat voor creator ben jij?", [
        "Algemeen", "⚽ Sport / Voetbal", "🏋️ Gym / Fitness",
        "🍕 Food", "🎮 Gaming", "🎵 Muziek", "😂 Humor / Comedy",
        "✨ Lifestyle / Vlog"
    ])
    st.markdown("---")
    st.markdown("### 📈 Trending in jouw niche")
    if st.button("🔍 Bekijk trends"):
        with st.spinner("AI zoekt trends..."):
            categorie = profiel.split(" ")[-1] if profiel != "Algemeen" else "general"
            trends = get_trends(categorie, taal)
            for regel in trends.split('\n'):
                if regel.startswith("TREND"):
                    inhoud = regel.split(":", 1)[-1].strip()
                    st.markdown(f"🔥 {inhoud}")

tab1, tab2, tab3 = st.tabs(["📝 Caption" if taal == "English" else "📝 Caption invoeren",
                              "🎬 Video",
                              "⚔️ Vergelijker" if taal == "Nederlands" else "⚔️ Compare"])

with tab1:
    label_caption = "Your caption" if taal == "English" else "Jouw caption"
    label_hashtags = "Hashtags" if taal == "English" else "Hashtags"
    caption = st.text_input(f"📝 {label_caption}", placeholder="Bijv: Morning routine check ☀️")
    hashtags = st.text_input(f"# {label_hashtags}", placeholder="#lifestyle #viral #morning")

    if st.button("🔍 Voorspel" if taal == "Nederlands" else "🔍 Predict", key="btn1"):
        if caption == "" and hashtags == "":
            st.warning("Vul eerst een caption in!" if taal == "Nederlands" else "Please fill in a caption first!")
        else:
            pred, kansen, klassen = voorspel(caption, hashtags)
            toon_resultaat(pred, kansen, klassen, caption, hashtags, taal=taal)

with tab2:
    st.write("Upload je video en beschrijf wat erin gebeurt." if taal == "Nederlands" else "Upload your video and describe what's in it.")
    video_file = st.file_uploader("🎬 Upload video", type=['mp4', 'mov', 'avi'])
    beschrijving = st.text_input("📝 Beschrijving" if taal == "Nederlands" else "📝 Description",
                                  placeholder="Bijv: Denzel Dumfries transfer naar Real Madrid, grappige video")

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
            st.image(frame_rgb, caption="📸 Screenshot", width=400)

    if video_file and beschrijving:
        btn_label = "🤖 Genereer captions & voorspel" if taal == "Nederlands" else "🤖 Generate captions & predict"
        if st.button(btn_label, key="btn2"):
            with st.spinner("AI bedenkt 3 caption variaties..." if taal == "Nederlands" else "AI is generating 3 caption variations..."):
                try:
                    variaties = genereer_caption_variaties(beschrijving, taal)

                    st.subheader("✨ 3 Caption variaties:")
                    beste_score = 0
                    beste_idx = 0
                    scores = []
                    for i, (stijl, cap_tekst, hash_tekst) in enumerate(variaties):
                        pred, kansen, klassen = voorspel(cap_tekst, hash_tekst)
                        score = bereken_score(pred, kansen, klassen)
                        scores.append((pred, kansen, klassen, score))
                        if score > beste_score:
                            beste_score = score
                            beste_idx = i

                    for i, (stijl, cap_tekst, hash_tekst) in enumerate(variaties):
                        pred, kansen, klassen, score = scores[i]
                        winnaar = " 🏆 Beste!" if i == beste_idx else ""
                        with st.expander(f"{stijl}{winnaar} — Score: {score}/100"):
                            st.success(f"**Caption:** {cap_tekst}")
                            st.info(f"**Hashtags:** {hash_tekst}")
                            toon_score_balk(score)

                    best_stijl, best_cap, best_hash = variaties[beste_idx]
                    best_pred = scores[beste_idx][0]

                    st.divider()
                    st.subheader("🎬 Video analyse")
                    with st.spinner("AI analyseert..."):
                        analyse = analyseer_video(beschrijving, best_pred, taal)
                        for regel in analyse.split('\n'):
                            if regel.startswith("WAAROM:"):
                                st.info(f"**📊 Waarom:** {regel.replace('WAAROM:', '').strip()}")
                            elif regel.startswith("STERK:"):
                                st.success(f"**💪 Sterk:** {regel.replace('STERK:', '').strip()}")
                            elif regel.startswith("VERBETER:"):
                                st.warning(f"**🔧 Verbeter:** {regel.replace('VERBETER:', '').strip()}")
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
    st.write("Test twee captions tegen elkaar!" if taal == "Nederlands" else "Test two captions against each other!")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Caption A")
        caption_a = st.text_input("📝 Caption A", placeholder="Deze goal is van een andere wereld ⚽🔥", key="ca")
        hashtags_a = st.text_input("# Hashtags A", placeholder="#voetbal #viral #fyp", key="ha")

    with col2:
        st.subheader("Caption B")
        caption_b = st.text_input("📝 Caption B", placeholder="POV: je scoort in de laatste minuut 😱", key="cb")
        hashtags_b = st.text_input("# Hashtags B", placeholder="#football #goals #trending", key="hb")

    btn_label3 = "⚔️ Vergelijk" if taal == "Nederlands" else "⚔️ Compare"
    if st.button(btn_label3, key="btn3"):
        if caption_a and caption_b:
            v_a, k_a, kl_a = voorspel(caption_a, hashtags_a)
            v_b, k_b, kl_b = voorspel(caption_b, hashtags_b)
            score_a = bereken_score(v_a, k_a, kl_a)
            score_b = bereken_score(v_b, k_b, kl_b)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Resultaat A")
                toon_score_balk(score_a)
                toon_resultaat(v_a, k_a, kl_a, caption_a, hashtags_a, toon_uitleg=False, taal=taal)
            with col2:
                st.subheader("Resultaat B")
                toon_score_balk(score_b)
                toon_resultaat(v_b, k_b, kl_b, caption_b, hashtags_b, toon_uitleg=False, taal=taal)

            st.divider()
            if score_a > score_b:
                st.success(f"🏆 Caption A wint! ({score_a} vs {score_b})")
            elif score_b > score_a:
                st.success(f"🏆 Caption B wint! ({score_b} vs {score_a})")
            else:
                st.info("🤝 Gelijkspel!")
        else:
            st.warning("Vul beide captions in!" if taal == "Nederlands" else "Fill in both captions!")
