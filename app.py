import streamlit as st
import requests
import spacy
import pandas as pd
from titlecase import titlecase
from sentence_transformers import SentenceTransformer, util
import torch
import os

# --- Tiêu đề và icon website ---
st.set_page_config(page_title="Q&A System as a Chatbot | KGQA", page_icon="🤖")

if 'answer' not in st.session_state:
    st.session_state.answer = None
    st.session_state.entity = None
    st.session_state.selected_uri = None
    st.session_state.last_query = ""

# --- load models ---
@st.cache_resource
def load_models():
    try:
        nlp_model = spacy.load("en_core_web_sm")
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        nlp_model = spacy.load("en_core_web_sm")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    return nlp_model, sbert_model

nlp, model = load_models()
DBPEDIA_URL = "https://dbpedia.org/sparql"

# --- Hàm lấy ra 'thực thể' từ câu hỏi ---
def improved_extract_info(question):
    clean_q = question.replace("?", "").strip()
    doc = nlp(clean_q)
    tokens = [t.text for t in doc]
    tokens_lower = [t.text.lower() for t in doc]
    entity_raw = ""
    if "of" in tokens_lower:
        idx = tokens_lower.index("of")
        entity_raw = " ".join(tokens[idx+1:])
    elif "the" in tokens_lower:
        idx = tokens_lower.index("the")
        entity_raw = " ".join(tokens[idx:])
    else:
        words = clean_q.split()
        proper_noun_parts = [w for w in words[2:] if w and w[0].isupper()] 
        
        if proper_noun_parts:
            entity_raw = " ".join(proper_noun_parts)
        elif doc.ents: 
            entity_raw = doc.ents[0].text
        else:
            chunks = list(doc.noun_chunks)
            if chunks: 
                entity_raw = chunks[-1].text
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
        response = requests.get(DBPEDIA_URL, params={'query': query, 'format': 'json'}, timeout=10)
        data = response.json()
        return [{"uri": b["p"]["value"], "label": b["label"]["value"]} for b in data["results"]["bindings"]]
    except: return []

def find_best_relation(question, properties, entity_name):
    if not properties: return None
    
    clean_entity = entity_name.replace("_", " ")
    relation_question = question.lower().replace(clean_entity.lower(), "").strip()
    
    labels = [p['label'] for p in properties]
    question_vec = model.encode(relation_question) # Dùng câu hỏi đã lọc thực thể
    label_vecs = model.encode(labels)
    sims = util.cos_sim(question_vec, label_vecs)[0]

    q_low = relation_question.lower()
    
    for i, p in enumerate(properties):
        l_low = p['label'].lower()
        
        # Mở rộng logic boost cho cả 'wife', 'spouse', 'husband'
        rel_keywords = ["wife", "husband", "spouse", "partner"]
        if any(w in q_low for w in rel_keywords) and any(w in l_low for w in rel_keywords):
            sims[i] += 0.35 # Tăng mức boost để đảm vượt ngưỡng
            
    best_idx = sims.argmax().item() 
    threshold = 0.2  # Có thể hạ thấp ngưỡng một chút nếu cần
    return properties[best_idx]['uri'] if sims[best_idx] > threshold else None

def execute_sparql_query(question, entity):
    if not entity: return "Không tìm thấy thực thể.", "None"
    available_props = get_all_properties(entity)
    if not available_props: return f"Không có dữ liệu cho {entity}. Hoặc do chưa xử lý được hết các trường hợp, hãy thông cảm...", "None"
    best_uri = find_best_relation(question, available_props, entity)
    if not best_uri: return "AI không khớp được quan hệ. Hoặc do chưa xử lý được hết các trường hợp, hãy thông cảm...", "None"
    display_uri = best_uri.replace("http://dbpedia.org/ontology/", "dbo:").replace("http://dbpedia.org/property/", "dbp:")
    q_low = question.lower()
    is_default_born = ("born" in q_low or "birth" in q_low) and not ("when" in q_low or "where" in q_low)
    if is_default_born:
        query = f"PREFIX dbr: <http://dbpedia.org/resource/> PREFIX dbo: <http://dbpedia.org/ontology/> SELECT DISTINCT ?res WHERE {{ {{ dbr:{entity} dbo:birthDate ?res . }} UNION {{ dbr:{entity} dbo:birthPlace ?res . }} UNION {{ dbr:{entity} dbo:wikiPageRedirects ?red . ?red dbo:birthDate ?res . }} UNION {{ dbr:{entity} dbo:wikiPageRedirects ?red . ?red dbo:birthPlace ?res . }} }}"
        display_uri = "dbo:birthDate/Place"
    else:
        query = f"PREFIX dbr: <http://dbpedia.org/resource/> PREFIX dbo: <http://dbpedia.org/ontology/> SELECT DISTINCT ?res WHERE {{ {{ dbr:{entity} <{best_uri}> ?res . }} UNION {{ dbr:{entity} dbo:wikiPageRedirects ?red . ?red <{best_uri}> ?res . }} FILTER (!isLiteral(?res) || lang(?res) = '' || lang(?res) = 'en') }}"
    try:
        data = requests.get(DBPEDIA_URL, params={'query': query, 'format': 'json'}, timeout=15).json()
        bindings = data["results"]["bindings"]
        if bindings:
            res_list = [b["res"]["value"].split('/')[-1].replace('_', ' ') if b["res"]["type"] == "uri" else b["res"]["value"] for b in bindings]
            final_result = ", ".join(list(set(res_list)))
            if len(final_result) > 500:
                final_result = final_result[:500] + "..."
                
            return final_result, display_uri
        return "Dữ liệu trống.", display_uri
    except Exception as e: return f"Lỗi: {str(e)}", "None"

# --- giao diện trên streamlit ---
st.title("🤖 Knowledge Graph QA System")
st.caption("Đề tài nhóm: Nguyễn Tuấn Đạt - Mai Thanh Duy - Châu Hải Đăng")

# Sử dụng st.form để Enter hoạt động như ý muốn
with st.form("qa_form", clear_on_submit=False):
    user_q = st.text_input("Nhập câu hỏi của bạn (Tiếng Anh):", placeholder="Who is the wife of Barack Obama?")
    submit_button = st.form_submit_button("Thực hiện truy vấn")

# Xử lý khi nhấn nút hoặc nhấn Enter
if submit_button and user_q:
    with st.status("Hệ thống đang xử lý...", expanded=True) as status:
        st.write("1. Đang trích xuất thực thể...")
        entity = improved_extract_info(user_q)
        
        if entity:
            st.write(f"2. Đang tìm quan hệ cho thực thể: **{entity}**...")
            answer, selected_uri = execute_sparql_query(user_q, entity)
            
            # Lưu vào session state
            st.session_state.answer = answer
            st.session_state.entity = entity
            st.session_state.selected_uri = selected_uri
            st.session_state.last_query = user_q
            
            status.update(label="Hoàn tất!", state="complete", expanded=False)
        else:
            status.update(label="Lỗi!", state="error")
            st.error("Không nhận diện được thực thể.")

# Hiển thị kết quả
if st.session_state.answer:
    st.markdown(f"**Câu hỏi:** {st.session_state.last_query}")
    st.success(f"**Kết quả:** {st.session_state.answer}")
    
    with st.expander("Chi tiết kỹ thuật"):
        st.write(f"- **Thực thể (Entity):** `{st.session_state.entity}`")
        st.write(f"- **Quan hệ (Property):** `{st.session_state.selected_uri}`")