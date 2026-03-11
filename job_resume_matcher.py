import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import pandas as pd

st.set_page_config(page_title="AI Resume Matcher", page_icon="💼")

st.title("💼 AI Resume Matcher")
st.write("Analyze how well your resume matches a job description")

resume_text = st.text_area("Paste Resume Text", height=200)
job_text = st.text_area("Paste Job Description", height=200)

def extract_keywords(text, n=10):
    words = text.lower().split()
    word_counts = Counter(words)
    common_words = word_counts.most_common(n)
    return [word[0] for word in common_words]

if st.button("Analyze Resume"):

    if resume_text and job_text:

        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform([resume_text, job_text])
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

        score = round(similarity * 100, 2)

        st.subheader("📊 Match Score")
        st.progress(int(score))
        st.success(f"{score}% match with job description")

        job_keywords = extract_keywords(job_text, 10)
        resume_words = resume_text.lower().split()

        matched = [word for word in job_keywords if word in resume_words]
        missing = [word for word in job_keywords if word not in resume_words]

        st.subheader("✅ Matched Skills")
        st.write(", ".join(matched) if matched else "No strong matches")

        st.subheader("⚠️ Missing Skills")
        st.write(", ".join(missing) if missing else "Your resume matches most skills!")

        # Skill comparison chart
        data = {
            "Skills": job_keywords,
            "Present in Resume": [1 if word in resume_words else 0 for word in job_keywords]
        }

        df = pd.DataFrame(data)
        st.subheader("📈 Skill Match Chart")
        st.bar_chart(df.set_index("Skills"))

    else:
        st.warning("Please paste both resume and job description.")
