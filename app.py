import streamlit as st
import requests
import spacy
import pandas as pd
from titlecase import titlecase
from sentence_transformers import SentenceTransformer, util
import torch
import os
# Cách gọi model an toàn trên Cloud sau khi đã khai báo trong requirements
@st.cache_resource
def load_nlp():
    # Load trực tiếp như một package
    return spacy.load("en_core_web_sm")

nlp = load_nlp()
# --- TỰ ĐỘNG TẢI MODEL NLP KHI DEPLOY ---
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- LOAD MODEL AI (Dùng Cache để tối ưu RAM 1GB của Streamlit Cloud) ---
@st.cache_resource
def load_kgqa_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

model = load_kgqa_model()
DBPEDIA_URL = "https://dbpedia.org/sparql"

def improved_extract_info(question):
    clean_q = question.replace("?", "").strip()
    doc = nlp(clean_q)
    tokens = [t.text for t in doc]
    tokens_lower = [t.text.lower() for t in doc]
    entity_raw = ""
    # Chiến thuật Greedy: Lấy toàn bộ sau 'of' hoặc 'the'
    if "of" in tokens_lower:
        idx = tokens_lower.index("of")
        entity_raw = " ".join(tokens[idx+1:])
    elif "the" in tokens_lower:
        idx = tokens_lower.index("the")
        entity_raw = " ".join(tokens[idx:])
    else:
        if doc.ents: entity_raw = doc.ents[0].text
        else:
            chunks = list(doc.noun_chunks)
            if chunks: entity_raw = chunks[-1].text
    if entity_raw:
        normalized = titlecase(entity_raw)
        return normalized.replace(" ", "_")
    return None

def get_all_properties(entity):
    query = f"""
    PREFIX dbr: <http://dbpedia.org/resource/>
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?p ?label WHERE {{
        {{ dbr:{entity} ?p ?o . }}
        UNION
        {{ dbr:{entity} dbo:wikiPageRedirects ?red . ?red ?p ?o . }}
        ?p rdfs:label ?label .
        FILTER (lang(?label) = 'en')
    }}
    """
    try:
        res = requests.get(DBPEDIA_URL, params={'query': query, 'format': 'json'}, timeout=10).json()
        return [{"uri": b["p"]["value"], "label": b["label"]["value"]} for b in res["results"]["bindings"]]
    except: return []

def find_best_relation(question, properties):
    if not properties: return None
    labels = [p['label'] for p in properties]
    q_vec = model.encode(question)
    l_vecs = model.encode(labels)
    sims = util.cos_sim(q_vec, l_vecs)[0]
    q_low = question.lower()
    is_born_query = "born" in q_low or "birth" in q_low
    for i, p in enumerate(properties):
        l_low = p['label'].lower()
        if "when" in q_low or "date" in q_low:
            if "date" in l_low: sims[i] += 0.3
        elif "where" in q_low or "place" in q_low:
            if "place" in l_low: sims[i] += 0.3
        elif is_born_query:
            if "date" in l_low or "place" in l_low: sims[i] += 0.2
        if any(w in q_low for w in ["wife", "husband", "spouse"]) and "spouse" in l_low: sims[i] += 0.3
        if "population" in q_low and any(w in l_low for w in ["as of", "density", "rank"]): sims[i] -= 0.5
    best_idx = sims.argmax().item()
    return properties[best_idx]['uri'] if sims[best_idx] > 0.25 else None

def improved_build_and_execute(question, entity):
    props = get_all_properties(entity)
    best_uri = find_best_relation(question, props)
    if not best_uri: return "AI không khớp được quan hệ.", "None"
    q_low = question.lower()
    is_default_born = ("born" in q_low or "birth" in q_low) and not ("when" in q_low or "where" in q_low)
    if is_default_born:
        query = f"PREFIX dbr: <http://dbpedia.org/resource/> PREFIX dbo: <http://dbpedia.org/ontology/> SELECT DISTINCT ?res WHERE {{ {{ dbr:{entity} dbo:birthDate ?res . }} UNION {{ dbr:{entity} dbo:birthPlace ?res . }} UNION {{ dbr:{entity} dbo:wikiPageRedirects ?red . ?red dbo:birthDate ?res . }} UNION {{ dbr:{entity} dbo:wikiPageRedirects ?red . ?red dbo:birthPlace ?res . }} }}"
    else:
        query = f"SELECT DISTINCT ?res WHERE {{ <http://dbpedia.org/resource/{entity}> <{best_uri}> ?res . FILTER (!isLiteral(?res) || lang(?res) = '' || lang(?res) = 'en') }}"
    try:
        data = requests.get(DBPEDIA_URL, params={'query': query, 'format': 'json'}, timeout=15).json()
        bindings = data["results"]["bindings"]
        if bindings:
            res_list = [b["res"]["value"].split('/')[-1].replace('_', ' ') if b["res"]["type"] == "uri" else b["res"]["value"] for b in bindings]
            return ", ".join(list(set(res_list))), best_uri
        return "Dữ liệu trống.", best_uri
    except: return "Lỗi truy vấn.", "None"

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="sori.io.vn | KGQA", page_icon="🤖")
st.title("Knowledge Graph QA System")
st.caption("Postdoc Project | Powered by DBpedia & S-BERT")

user_q = st.text_input("Nhập câu hỏi của bạn:", placeholder="Who is the wife of John Lennon?")

if st.button("Truy vấn"):
    if user_q:
        with st.status("Hệ thống đang phân tích...", expanded=True) as status:
            st.write("Đang bốc thực thể...")
            entity = improved_extract_info(user_q)
            if entity:
                st.write(f"AI đang tìm quan hệ cho: **{entity}**...")
                answer, uri = improved_build_and_execute(user_q, entity)
                status.update(label="Hoàn tất!", state="complete", expanded=False)
                st.subheader("Đáp án:")
                st.success(answer)
                with st.expander("Thông tin kỹ thuật"):
                    st.write(f"**Entity:** `{entity}`")
                    st.write(f"**Property URI:** `{uri}`")
            else:
                status.update(label="Lỗi!", state="error")
                st.error("Không nhận diện được thực thể.")