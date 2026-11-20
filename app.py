"""Class Support Dashboard - modified dataset edition (Streamlit, 5 tabs)

    streamlit run app.py

Same models and rules as the original app, new interface. The app ONLY loads saved
models and predicts; it never retrains after a click. Models are trained in the Colab notebook.
"""
import altair as alt
import pandas as pd
import streamlit as st

from src import clustering, data_loader as dl, llm, nlp, tabular, validation
from src.config import MAX_COMMENT_CHARS, MODELS, REPORTS, UNITS

st.set_page_config(page_title="Class Support Dashboard", page_icon="🧭", layout="wide")

TEAL, AMBER, SLATE, ROSE = "#0F766E", "#D97706", "#475569", "#BE123C"
PALETTE = [TEAL, AMBER, "#6366F1", ROSE, SLATE]

st.markdown(f"""
<style>
.block-container {{padding-top: 3.2rem; max-width: 1250px;}}
.hero {{background: linear-gradient(120deg, {TEAL} 0%, #134E4A 60%, #1E293B 100%);
        color: #F8FAFC; border-radius: 18px; padding: 26px 30px; margin-bottom: 14px;}}
.hero h1 {{color: #F8FAFC; font-size: 2.0rem; margin: 0 0 4px 0; padding: 0;}}
.hero p {{color: #CCFBF1; margin: 0; font-size: 0.98rem;}}
.pill {{display: inline-block; padding: 3px 11px; margin: 10px 6px 0 0; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; background: rgba(255,255,255,0.14); color: #F0FDFA;
        border: 1px solid rgba(255,255,255,0.25);}}
.kpi {{background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 5px solid {TEAL};
       border-radius: 14px; padding: 14px 16px; box-shadow: 0 1px 2px rgba(15,23,42,0.05);}}
.kpi .label {{color: #64748B; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .05em;}}
.kpi .value {{color: #0F172A; font-size: 1.55rem; font-weight: 700; margin-top: 2px;}}
.kpi .sub {{color: #94A3B8; font-size: 0.78rem;}}
.result {{border-radius: 14px; padding: 16px 20px; margin: 8px 0; border: 1px solid #E2E8F0;}}
.result.flag {{background: #FFF1F2; border-left: 6px solid {ROSE};}}
.result.ok {{background: #F0FDFA; border-left: 6px solid {TEAL};}}
.result .big {{font-size: 1.25rem; font-weight: 700; color: #0F172A;}}
.groupcard {{background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 5px solid var(--c);
             border-radius: 14px; padding: 14px 16px; height: 100%;}}
.groupcard h4 {{margin: 0 0 6px 0; color: #0F172A;}}
.groupcard .n {{font-size: 1.6rem; font-weight: 700; color: var(--c);}}
.section-note {{color: #64748B; font-size: 0.86rem;}}
.stTabs [data-baseweb="tab"] {{font-weight: 600;}}
</style>
""", unsafe_allow_html=True)

ss = st.session_state
for k in ("prediction", "group", "digit", "comments", "ai_summary", "ai_info", "prompt_log"):
    ss.setdefault(k, None)


# ======================================================== cached loaders
@st.cache_data(max_entries=1)
def get_data(stamp):
    return dl.load_main(), dl.load_meta()


def file_stamp(path):
    """Size + modified time of a model file, used as the cache key so new model files load automatically."""
    try:
        s = path.stat()
        return f"{s.st_size}-{s.st_mtime_ns}"
    except FileNotFoundError:
        return "missing"


@st.cache_resource(max_entries=1)
def get_tree(stamp):
    return tabular.load_model() if tabular.MODEL_PATH.exists() else None


@st.cache_resource(max_entries=1)
def get_clusters(stamp):
    return clustering.load_model() if clustering.MODEL_PATH.exists() else None


@st.cache_resource(max_entries=1)
def get_cnn(stamp):
    from src import image_model
    return image_model.load_model()


def describe_cnn(model):
    parts = []
    for layer in model.layers:
        name = layer.__class__.__name__
        if name == "Conv2D":
            parts.append(f"Conv2D({layer.filters})")
        elif name == "Dense":
            parts.append(f"Dense({layer.units})")
    return " → ".join(parts) + f" · {model.count_params():,} parameters"


@st.cache_data(max_entries=1)
def get_feedback_analysis(stamp):
    return nlp.analyse(dl.load_feedback())


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def bar(df, x, y, x_title, y_title, color=TEAL, sort=None):
    return alt.Chart(df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color=color).encode(
        x=alt.X(x, title=x_title, sort=sort, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(y, title=y_title)).properties(height=230)


try:
    main_df, meta = get_data(file_stamp(dl.DATA / "main.csv") + file_stamp(dl.DATA / "meta.json"))
    feedback_ok = (dl.DATA / "feedback.csv").exists() and (dl.DATA / "topic_keywords.json").exists()
except FileNotFoundError as e:
    st.error(f"Data file missing: `{e.filename}`. Copy the supplied files into the `data/` folder. See README.md.")
    st.stop()

num, cat, features, target, cluster_features = dl.feature_lists(meta)
schema = validation.build_schema(meta, main_df)
train_df, val_df, test_df = dl.get_splits(main_df)
tree = get_tree(file_stamp(tabular.MODEL_PATH))
kmeans = get_clusters(file_stamp(clustering.MODEL_PATH))

edition = "Modified dataset edition" if "MODIFIED" in str(meta.get("title", "")).upper() else "Classroom edition"
st.markdown(f"""
<div class="hero">
  <h1>🧭 Class Support Dashboard</h1>
  <p>Marks, learning profiles, rating cards and course comments in one place, for a teacher's review
     after the second assessment.</p>
  <span class="pill">{edition}</span><span class="pill">Decision support only</span>
  <span class="pill">Works offline</span><span class="pill">{len(main_df)} records</span>
</div>""", unsafe_allow_html=True)

if tree is None or kmeans is None:
    st.warning("Trained models not found in `models/`. Run the Colab notebook (Parts 2-4) and copy its `models/` folder here.")

MEANING = {1: "May need academic support - suggest a conversation for the teacher to review",
           0: "No support flag from the model"}

fb_base, fb_error = None, None
if feedback_ok:
    try:
        fb_base = get_feedback_analysis(file_stamp(dl.DATA / "feedback.csv"))
    except Exception as e:
        fb_error = str(e)

tabs = st.tabs(["🏫 Class & prediction", "🧭 Learning profiles", "✍️ Rating digit",
                "🗨️ Comments", "📄 Teaching report"])


# ======================================================== TAB 1
with tabs[0]:
    c = st.columns(4)
    kpi(c[0], "Records", len(main_df), "fixed splits")
    kpi(c[1], "Train · Val · Test", f"{len(train_df)} · {len(val_df)} · {len(test_df)}", "rows")
    kpi(c[2], "Needs support", f"{train_df[target].mean():.0%}", "training rows")
    kpi(c[3], "Average G2", f"{train_df['G2'].mean():.1f}", "out of 20, training rows")
    st.write("")

    left, right = st.columns([5, 4], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("##### 📝 Check one student record")
            examples = dl.load_example_inputs()
            defaults = {}
            if examples is not None and len(examples):
                labels = [f"Example {i + 1}" + (f" · {r['record_id']}" if "record_id" in r else "")
                          for i, r in examples.iterrows()]
                pick = st.selectbox("Start from a supplied example record", ["(blank form)"] + labels)
                if pick != "(blank form)":
                    defaults = examples.iloc[labels.index(pick)].to_dict()

            values = {}
            with st.form("record", border=False):
                cols = st.columns(2)
                for i, col in enumerate(schema):
                    rule = schema[col]
                    with cols[i % 2]:
                        if rule["type"] == "cat":
                            opts = rule["allowed"]
                            d = str(defaults.get(col, opts[0]))
                            values[col] = st.selectbox(col, opts, index=opts.index(d) if d in opts else 0)
                        else:
                            lo, hi = rule["range"]
                            d = defaults.get(col, train_df[col].median())
                            if rule.get("integer"):
                                d = int(round(float(d)))
                            values[col] = st.number_input(f"{col} ({lo:g}-{hi:g})", value=float(d), step=1.0,
                                                          help=UNITS.get(col, ""), format="%g")
                go = st.form_submit_button("Check support need", type="primary")

    with right:
        st.markdown("##### 📌 Result")
        if go:
            clean, errors = validation.validate_record(values, schema)
            if errors:
                st.error("Record rejected - nothing was sent to the model:\n\n" + "\n".join(f"- {e}" for e in errors))
            elif tree is None:
                st.error("Decision tree model is missing (`models/supervised.joblib`).")
            else:
                label, p = tabular.predict_one(tree, clean, features)
                ss.prediction = {"label": label, "meaning": MEANING[label],
                                 "probability": round(p, 3), "G1": clean["G1"], "G2": clean["G2"]}
                if kmeans is not None:
                    g = clustering.predict_group(kmeans, clean, cluster_features)
                    ss.group = {"group": g, "description": clustering.load_descriptions().get(str(g), "no description saved")}
                ss.ai_summary = None

        if ss.prediction:
            p = ss.prediction
            css = "flag" if p["label"] == 1 else "ok"
            icon = "🚩" if p["label"] == 1 else "✅"
            st.markdown(f'<div class="result {css}"><div class="big">{icon} {p["meaning"]}</div>'
                        f'<div class="section-note">G1 = {p["G1"]}/20 · G2 = {p["G2"]}/20</div></div>',
                        unsafe_allow_html=True)
            st.caption("Share of similar training records with label 1 in this tree leaf (not a calibrated certainty)")
            st.progress(float(p["probability"]), text=f"{p['probability']:.0%}")
            box = st.error if p["label"] == 1 else st.success
            box(f"**Model output: needs_support = {p['label']}** — teacher review required.")
            if ss.group:
                st.info(f"**Learning profile {ss.group['group']}:** {ss.group['description']}")
        else:
            st.markdown('<div class="result ok"><div class="section-note">Choose an example or fill in the form, '
                        'then press <b>Check support need</b>.</div></div>', unsafe_allow_html=True)

    st.write("")
    with st.container(border=True):
        st.markdown("##### 📊 Class overview (training rows only)")
        ch = st.columns(3)
        counts = train_df[target].value_counts().sort_index().rename_axis(target).reset_index(name="records")
        counts[target] = counts[target].map({0: "0 · no", 1: "1 · yes"})
        ch[0].altair_chart(bar(counts, f"{target}:N", "records:Q", "needs_support", "records")
                           .encode(color=alt.Color(f"{target}:N", scale=alt.Scale(range=[TEAL, ROSE]), legend=None)),
                           width="stretch")
        ch[1].altair_chart(alt.Chart(train_df).mark_area(color=TEAL, opacity=0.35, line={"color": TEAL})
                           .encode(x=alt.X("G2:Q", bin=alt.Bin(step=2, extent=[0, 20]), title="G2 (0-20)"),
                                   y=alt.Y("count():Q", title="records")).properties(height=230), width="stretch")
        avg = train_df.groupby("studytime")["G2"].mean().round(2).reset_index()
        ch[2].altair_chart(alt.Chart(avg).mark_line(point=alt.OverlayMarkDef(size=90, color=AMBER), color=AMBER,
                                                    strokeWidth=3)
                           .encode(x=alt.X("studytime:O", title="studytime", axis=alt.Axis(labelAngle=0)),
                                   y=alt.Y("G2:Q", title="average G2", scale=alt.Scale(zero=False)))
                           .properties(height=230), width="stretch")
        st.caption("Support labels · G2 distribution · average G2 by study time. Associations only, not causes.")

    metrics_csv = REPORTS / "tabular_test_metrics.csv"
    if metrics_csv.exists():
        with st.expander("🧪 Model test results (decision tree vs baseline)"):
            st.dataframe(pd.read_csv(metrics_csv), width="stretch", hide_index=True)
            img = REPORTS / "02_tabular_confusion.png"
            if img.exists():
                st.image(str(img), width=600)

    with st.expander("📂 Optional: check a CSV of records"):
        up = st.file_uploader("CSV with the feature columns", type="csv", key="batch")
        if up is not None and tree is not None:
            try:
                bdf = pd.read_csv(up)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                bdf = None
            if bdf is not None:
                errs = validation.validate_table(bdf, schema)
                if not errs:
                    errs = [f"Row {i + 1}: " + "; ".join(e) for i, r in bdf.iterrows()
                            for e in [validation.validate_record(r.to_dict(), schema)[1]] if e]
                if errs:
                    st.error("\n".join(f"- {e}" for e in errs[:20]))
                else:
                    st.dataframe(tabular.predict_table(tree, bdf, features), width="stretch", hide_index=True)


# ======================================================== TAB 2
with tabs[1]:
    st.markdown("##### 🧭 Learning profiles · KMeans with 3 groups")
    st.markdown(f'<span class="section-note">Inputs: {", ".join(cluster_features)}. Group numbers are arbitrary; '
                'a group is a profile, not a predicted class or a judgement of ability.</span>', unsafe_allow_html=True)
    if kmeans is None:
        st.info("Run the Colab notebook (Part 3) to create `models/clusters.joblib`.")
    else:
        summary, groups = clustering.profile_table(kmeans, train_df, cluster_features)
        desc = clustering.load_descriptions()
        cols = st.columns(len(summary))
        for col, (g, row) in zip(cols, summary.iterrows()):
            colour = PALETTE[int(g) % len(PALETTE)]
            mine = " ⭐" if ss.group and ss.group["group"] == int(g) else ""
            col.markdown(
                f'<div class="groupcard" style="--c:{colour}"><h4>Profile {g}{mine}</h4>'
                f'<div class="n">{int(row["size"])}</div><div class="section-note">students in training rows</div>'
                f'<p style="margin-top:8px">G1 {row["G1"]} · G2 {row["G2"]} · study {row["studytime"]} · '
                f'failures {row["failures"]}</p><div class="section-note">{desc.get(str(g), "")}</div></div>',
                unsafe_allow_html=True)
        st.write("")
        plot = train_df[cluster_features].copy()
        plot["profile"] = groups.astype(str)
        layers = [alt.Chart(plot).mark_circle(size=70, opacity=.55).encode(
            x=alt.X("G1:Q", title="G1 (0-20)"), y=alt.Y("G2:Q", title="G2 (0-20)"),
            color=alt.Color("profile:N", scale=alt.Scale(range=PALETTE[:len(summary)]), title="Profile"),
            tooltip=cluster_features + ["profile"])]
        if ss.prediction:
            me = pd.DataFrame([{"G1": ss.prediction["G1"], "G2": ss.prediction["G2"]}])
            layers.append(alt.Chart(me).mark_point(shape="diamond", size=380, color="#0F172A", filled=True)
                          .encode(x="G1:Q", y="G2:Q"))
        with st.container(border=True):
            st.altair_chart(alt.layer(*layers).properties(height=380), width="stretch")
            st.caption("◆ = the record checked in the first tab (its profile card has a ⭐). Overlapping points are normal.")
        with st.expander("Table: group sizes and averages (original units)"):
            st.dataframe(summary, width="stretch")


# ======================================================== TAB 3
with tabs[2]:
    st.markdown("##### ✍️ Rating-card digit reader")
    st.markdown('<span class="section-note">Reads ONE digit from a demonstration rating card; the teacher confirms it. '
                'Trained on MNIST - never use it to assign an academic mark.</span>', unsafe_allow_html=True)
    if not (MODELS / "images.keras").exists():
        st.info("CNN model not found. Run the Colab notebook (Part 4) to create `models/images.keras`.")
    else:
        try:
            from PIL import Image
            from src import image_model
            cnn = get_cnn(file_stamp(image_model.MODEL_PATH))
        except ImportError:
            cnn = None
            st.error("TensorFlow is not installed. Keep `tensorflow-cpu` in requirements.txt.")
        except Exception as e:
            cnn = None
            st.error(f"Could not load the CNN: {e}")
        if cnn is not None:
            st.caption(f"Model in use: {describe_cnn(cnn)}")
            classes = [str(c) for c in meta.get("image_classes", list(range(10)))]
            left, right = st.columns([2, 3], gap="large")
            with left:
                with st.container(border=True):
                    samples = dl.list_sample_images()
                    sources = (["Supplied sample image"] if samples else []) + ["Upload PNG/JPG"]
                    src_choice = st.radio("Image source", sources, horizontal=True)
                    pil = None
                    if src_choice == "Supplied sample image":
                        name = st.selectbox("Sample", [p.name for p in samples])
                        pil = Image.open(dl.SAMPLE_IMAGES / name)
                    else:
                        up = st.file_uploader("One digit per image", type=["png", "jpg", "jpeg"], key="digit_upload")
                        if up:
                            try:
                                pil = Image.open(up)
                            except Exception:
                                st.error("This file is not a readable image.")
                    if pil is not None:
                        inv = st.selectbox("Colour handling", ["auto", "invert", "keep"],
                                           help="MNIST is a white digit on black. 'auto' inverts light backgrounds.")
                        arr = image_model.preprocess(pil, {"auto": "auto", "invert": True, "keep": False}[inv])
                        a, b = st.columns(2)
                        a.image(pil, width=130, caption="Original")
                        b.image(arr[0].squeeze().astype("uint8"), width=130, caption="Model input 28×28")
            with right:
                if pil is not None:
                    k, scores = image_model.predict(cnn, arr)
                    with st.container(border=True):
                        m1, m2 = st.columns(2)
                        m1.metric("Predicted digit", classes[k], f"score {scores[k]:.0%}")
                        second = scores.argsort()[-2]
                        m2.metric("Runner-up", classes[second], f"score {scores[second]:.0%}", delta_color="off")
                        sdf = pd.DataFrame({"digit": classes, "score": scores})
                        sdf["top"] = sdf["digit"] == classes[k]
                        st.altair_chart(alt.Chart(sdf).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                            x=alt.X("digit:N", axis=alt.Axis(labelAngle=0)), y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1])),
                            color=alt.condition("datum.top", alt.value(TEAL), alt.value("#CBD5E1")))
                            .properties(height=200), width="stretch")
                        st.caption("A softmax score is not a calibrated probability and the model cannot say 'unknown'.")
                    with st.container(border=True):
                        confirmed = st.selectbox("Teacher confirmation - correct the digit if needed", classes, index=k)
                        if st.button("Confirm digit"):
                            ss.digit = int(confirmed)
                            ss.ai_summary = None
                        if ss.digit is not None:
                            st.success(f"Confirmed rating digit: **{ss.digit}**")
                else:
                    st.info("Pick a sample or upload an image on the left.")


# ======================================================== TAB 4
with tabs[3]:
    st.markdown("##### 🗨️ Course comments · VADER sentiment and keyword topics")
    if not feedback_ok:
        st.error("`data/feedback.csv` or `data/topic_keywords.json` is missing.")
    elif fb_error:
        st.error(fb_error)
    else:
        fb = fb_base
        up = st.file_uploader("Optional: analyse another CSV with a `text` column", type="csv", key="comments_csv")
        if up is not None:
            try:
                other = pd.read_csv(up)
                if "text" not in other.columns:
                    st.error("CSV needs a column named `text`. Showing the supplied comments instead.")
                else:
                    other["text"] = other["text"].astype("string").str.slice(0, MAX_COMMENT_CHARS)
                    fb = nlp.analyse(other)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
        ss.comments = fb

        sc, tc = nlp.sentiment_counts(fb), nlp.topic_counts(fb)
        k = st.columns(4)
        kpi(k[0], "Comments", sum(sc.values()), "non-empty")
        kpi(k[1], "Positive", sc.get("positive", 0))
        kpi(k[2], "Neutral", sc.get("neutral", 0))
        kpi(k[3], "Negative", sc.get("negative", 0))
        st.write("")
        c1, c2 = st.columns(2)
        with c1, st.container(border=True):
            st.markdown("**Sentiment share** (empty comments excluded)")
            sdf = pd.DataFrame({"sentiment": list(sc), "comments": list(sc.values())})
            st.altair_chart(alt.Chart(sdf).mark_arc(innerRadius=60).encode(
                theta="comments:Q", color=alt.Color("sentiment:N", scale=alt.Scale(
                    domain=["negative", "neutral", "positive"], range=[ROSE, "#94A3B8", TEAL])),
                tooltip=["sentiment", "comments"]).properties(height=240), width="stretch")
        with c2, st.container(border=True):
            st.markdown("**Topics**")
            tdf = pd.DataFrame({"topic": list(tc), "comments": list(tc.values())})
            st.altair_chart(alt.Chart(tdf).mark_bar(color=AMBER, cornerRadiusEnd=6).encode(
                y=alt.Y("topic:N", sort="-x", title=None), x=alt.X("comments:Q")).properties(height=240),
                width="stretch")

        if "sentiment_label" in fb.columns:
            from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
            ref, pred = fb["sentiment_label"], fb["predicted_sentiment"]
            m = st.columns(3)
            m[0].metric("Accuracy vs reference", f"{accuracy_score(ref, pred):.2f}")
            m[1].metric("Macro F1", f"{f1_score(ref, pred, average='macro', labels=nlp.LABELS, zero_division=0):.2f}")
            m[2].metric("Disagreements", int((ref != pred).sum()))
            with st.expander("Confusion matrix and disagreements"):
                st.dataframe(pd.DataFrame(confusion_matrix(ref, pred, labels=nlp.LABELS),
                                          index=[f"ref {l}" for l in nlp.LABELS],
                                          columns=[f"pred {l}" for l in nlp.LABELS]))
                st.dataframe(fb[ref != pred][["text", "sentiment_label", "predicted_sentiment"]],
                             width="stretch", hide_index=True)
        with st.expander("All comments"):
            st.dataframe(fb[[c for c in ("text", "sentiment_label", "predicted_sentiment", "topic", "matched_keyword")
                             if c in fb.columns]], width="stretch", hide_index=True)

        with st.container(border=True):
            st.markdown("**🔎 Try one comment**")
            mode = st.radio("Comment source", ["Pick a supplied comment", "Type a comment (optional demo)"], horizontal=True)
            if mode == "Pick a supplied comment":
                texts = fb["text"].fillna("").astype(str).tolist()
                i = st.selectbox("Comment", range(len(texts)), format_func=lambda j: texts[j][:110] or "(empty comment)")
                txt = texts[i]
            else:
                txt = st.text_area("English comment", max_chars=MAX_COMMENT_CHARS,
                                   help=f"Up to {MAX_COMMENT_CHARS} characters. English only.")
            if st.button("Analyse comment"):
                if nlp.is_empty(txt):
                    st.warning("This comment is empty. It is treated as **empty** and is not counted in sentiment totals.")
                else:
                    topic, kw = nlp.assign_topic(txt)
                    s = nlp.sentiment(txt)
                    colour = {"positive": TEAL, "negative": ROSE}.get(s, SLATE)
                    st.markdown(f'<div class="result" style="border-left:6px solid {colour}"><b>{s.upper()}</b> · '
                                f'compound {nlp.compound(txt):+.2f} · topic <b>{topic}</b>'
                                + (f' (keyword “{kw}”)' if kw else " (no keyword)") + "</div>", unsafe_allow_html=True)
        st.caption("Sentiment = tone, topic = subject. Negation, sarcasm and mixed opinions can be misread; "
                   "Urdu / Roman Urdu are outside this English baseline.")


# ======================================================== TAB 5
with tabs[4]:
    st.markdown("##### 📄 Teaching report")
    fbs = ss.comments
    snapshot = llm.build_snapshot(
        prediction=ss.prediction, group=ss.group, digit=ss.digit,
        sentiment_counts=nlp.sentiment_counts(fbs) if fbs is not None else None,
        topic_counts=nlp.topic_counts(fbs) if fbs is not None else None,
        example_comment=(fbs[fbs["predicted_sentiment"] == "negative"]["text"].head(1).tolist() or [None])[0]
        if fbs is not None else None)

    steps = [("Record checked", ss.prediction is not None), ("Profile found", ss.group is not None),
             ("Digit confirmed", ss.digit is not None), ("Comments analysed", fbs is not None)]
    sc = st.columns(4)
    for col, (name, done) in zip(sc, steps):
        kpi(col, name, "✔ done" if done else "— not yet")
    st.write("")

    mode = st.radio("Summary mode", ["Offline summary", "Groq AI summary (optional)"], horizontal=True)
    with st.container(border=True):
        if mode == "Offline summary":
            st.markdown(llm.offline_summary(snapshot))
            st.download_button("⬇ Download report (.md)", llm.offline_summary(snapshot), "teaching_summary.md")
        else:
            has_key = bool(llm.get_api_key())
            st.caption(f"Model: `{llm.get_model_id()}` · Groq Free Plan only · "
                       + ("key found" if has_key else "no key found - add GROQ_API_KEY in Secrets"))
            with st.expander("Data that will be sent (totals + one sanitised comment only)"):
                st.json(snapshot)
            if st.button("Generate AI summary", type="primary", disabled=not has_key):
                with st.spinner("Calling Groq once..."):
                    ss.ai_summary, ss.ai_info = llm.safe_groq_summary(snapshot)
            if ss.ai_summary:
                st.markdown(ss.ai_summary)
                st.caption(f"AI-generated by {ss.ai_info}. Check every number before using it.")
            elif ss.ai_info:
                st.warning(f"{ss.ai_info} The app is still fully usable - here is the Offline summary:")
                st.markdown(llm.offline_summary(snapshot))

    with st.expander("🧪 Prompt comparison (Section 8.2)"):
        cases = dl.load_prompt_cases()
        if not cases:
            st.info("`data/prompt_test_cases.json` not found.")
        else:
            st.caption(f"{len(cases)} supplied cases · vague vs structured prompt · at most 10 calls per team.")
            ci = st.selectbox("Case", range(len(cases)),
                              format_func=lambda j: str(cases[j].get("id", cases[j].get("case", f"Case {j + 1}")))
                              if isinstance(cases[j], dict) else f"Case {j + 1}")
            case = cases[ci]
            st.json(case)
            payload = case.get("snapshot", case.get("input", case)) if isinstance(case, dict) else case
            pstyle = st.radio("Prompt", list(llm.PROMPTS), horizontal=True)
            ss.setdefault("calls_made", 0)
            if st.button(f"Run on Groq ({ss.calls_made}/10 calls used)",
                         disabled=ss.calls_made >= 10 or not llm.get_api_key()):
                ss.calls_made += 1
                text, info = llm.safe_groq_summary(payload if isinstance(payload, dict) else {"case": payload},
                                                   llm.PROMPTS[pstyle])
                ss.prompt_log = (ss.prompt_log or []) + [{"case": ci + 1, "prompt": pstyle,
                                                          "source": info, "output": text or "(failed)"}]
            if ss.prompt_log:
                log = pd.DataFrame(ss.prompt_log)
                st.dataframe(log, width="stretch", hide_index=True)
                st.download_button("Download log (CSV)", log.to_csv(index=False), "prompt_comparison.csv")

st.markdown("---")
st.caption("Class Support Dashboard · educational prototype · a teacher must review every suggestion. "
           "The support flag is a classroom modelling target, not a decision about a real student.")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🧭 Dashboard")
    st.markdown(f"**{edition}**")
    st.markdown(f"Dataset: {meta.get('title', '')}")
    mod = meta.get("modification")
    if isinstance(mod, dict):
        with st.expander("What changed in this dataset"):
            for k2, v in mod.items():
                if k2 != "unchanged":
                    st.markdown(f"- **{k2}**: {v}")
    st.divider()
    st.markdown("**Models**")
    st.markdown(f"- Decision tree: {'✅' if tree is not None else '❌'}\n"
                f"- KMeans: {'✅' if kmeans is not None else '❌'}\n"
                f"- CNN: {'✅' if (MODELS / 'images.keras').exists() else '❌'}")
    if st.button("Reload models", help="Clear the cache and load the files in data/ and models/ again"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.caption("Decision Tree · KMeans · CNN · NLTK VADER · Groq (optional). Free tools only.")
