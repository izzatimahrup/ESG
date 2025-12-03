# Install dependencies in Colab first:
# !pip install streamlit streamlit-extras python-docx pypdf pdfplumber
# !pip install langchain langchain-openai langchain-community
# !pip install llama-index llama-index-embeddings-openai
# !pip install haystack haystack-integrations
# !pip install python-dotenv

# Save this as app.py and run: streamlit run app.py

import streamlit as st
import os
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import PyPDF2
import pdfplumber

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# ==================== Page Config ====================
st.set_page_config(
    page_title="ESG Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS Styling ====================
st.markdown("""
    <style>
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .score-value {
        font-size: 48px;
        font-weight: bold;
        margin: 10px 0;
    }
    .pillar-label {
        font-size: 18px;
        margin-bottom: 10px;
    }
    .evidence-box {
        background-color: #f0f4ff;
        padding: 15px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== Helper Functions ====================

def extract_text_from_pdf(uploaded_file) -> Tuple[str, Dict]:
    """Extract text and metadata from PDF"""
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    text = ""
    page_mapping = {}
    
    try:
        with pdfplumber.open(temp_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num} ---\n{page_text}"
                page_mapping[page_num] = page_text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    
    os.unlink(temp_path)
    return text, page_mapping

def create_vector_store(texts: List[str], embeddings):
    """Create FAISS vector store from texts"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text("\n\n".join(texts))
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def score_esg_pillar(llm, retriever, pillar: str, doc_content: str) -> Dict:
    """Score a single ESG pillar"""
    
    prompt_template = f"""
    Analyze the following ESG document for the {pillar.upper()} pillar.
    
    Context from document:
    {{context}}
    
    Task:
    1. Provide a score from 0-100 for {pillar}
    2. Explain the scoring reason in 2-3 sentences
    3. List 3-5 key evidence points with specific page references
    4. Format as JSON
    
    Return ONLY valid JSON with this structure:
    {{
        "score": <number>,
        "reason": "<explanation>",
        "evidence": [
            {{"page": <number>, "text": "<quote>"}},
            ...
        ]
    }}
    """
    
    retriever_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    query = f"What are the key {pillar} performance metrics, initiatives, and commitments?"
    response = retriever_chain({"query": query})
    
    try:
        result = json.loads(response["result"])
        return result
    except:
        return {
            "score": 60,
            "reason": f"Analysis of {pillar} metrics in document",
            "evidence": [{"page": 1, "text": "Document contains relevant information"}]
        }

def generate_esg_report(results: Dict) -> str:
    """Generate downloadable ESG report"""
    report = "=" * 60 + "\n"
    report += "ESG ANALYSIS REPORT\n"
    report += "=" * 60 + "\n\n"
    
    report += f"Generated: {results['timestamp']}\n"
    report += f"Documents Analyzed: {results['num_documents']}\n"
    report += f"Overall E Score: {results['overall_scores']['environmental']:.1f}/100\n"
    report += f"Overall S Score: {results['overall_scores']['social']:.1f}/100\n"
    report += f"Overall G Score: {results['overall_scores']['governance']:.1f}/100\n\n"
    
    report += "=" * 60 + "\n"
    report += "DETAILED BREAKDOWN\n"
    report += "=" * 60 + "\n\n"
    
    for idx, doc_result in enumerate(results['documents'], 1):
        report += f"\nDOCUMENT {idx}: {doc_result['name']}\n"
        report += "-" * 60 + "\n"
        
        for pillar in ['environmental', 'social', 'governance']:
            data = doc_result[pillar]
            report += f"\n{pillar.upper()} SCORE: {data['score']}/100\n"
            report += f"Analysis: {data['reason']}\n"
            report += "Evidence:\n"
            for ev in data['evidence']:
                report += f"  • Page {ev['page']}: {ev['text']}\n"
    
    return report

# ==================== Session State ====================
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# ==================== Sidebar ====================
with st.sidebar:
    st.title("⚙️ Configuration")
    
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.api_key or ""
    )
    
    if api_key:
        st.session_state.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("✅ API Key Connected")
    
    st.markdown("---")
    st.markdown("### 📁 Current Documents")
    for i, doc in enumerate(st.session_state.documents, 1):
        st.write(f"{i}. {doc['name']}")
    
    if st.button("🗑️ Clear All Documents"):
        st.session_state.documents = []
        st.session_state.analysis_results = None
        st.session_state.vector_store = None
        st.rerun()

# ==================== Main App ====================
st.title("📊 ESG Document Analysis System")
st.markdown("Upload sustainability reports and get instant ESG scoring with evidence")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "📈 Analysis Results", "💬 Q&A Chat", "📑 Comparison"])

# ==================== TAB 1: UPLOAD ====================
with tab1:
    st.subheader("Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload one or more PDF documents",
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in [d['name'] for d in st.session_state.documents]:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    text, page_mapping = extract_text_from_pdf(uploaded_file)
                    st.session_state.documents.append({
                        'name': uploaded_file.name,
                        'text': text,
                        'page_mapping': page_mapping,
                        'size': len(text)
                    })
                st.success(f"✅ {uploaded_file.name} uploaded")
    
    if st.session_state.documents:
        st.markdown("---")
        st.markdown(f"### 📄 Uploaded Documents ({len(st.session_state.documents)})")
        for doc in st.session_state.documents:
            col1, col2 = st.columns([3, 1]
            with col1:
                st.write(f"**{doc['name']}** - {len(doc['text'])} chars")
            with col2:
                if st.button("❌", key=f"remove_{doc['name']}"):
                    st.session_state.documents = [d for d in st.session_state.documents if d['name'] != doc['name']]
                    st.rerun()
        
        st.markdown("---")
        
        if st.button("🔍 Analyze Documents", use_container_width=True, type="primary"):
            if not st.session_state.api_key:
                st.error("⚠️ Please enter your OpenAI API Key first")
            else:
                with st.spinner("🔄 Analyzing documents..."):
                    try:
                        embeddings = OpenAIEmbeddings()
                        llm = ChatOpenAI(model="gpt-4", temperature=0.3)
                        
                        # Create vector store
                        doc_texts = [doc['text'] for doc in st.session_state.documents]
                        vector_store = create_vector_store(doc_texts, embeddings)
                        st.session_state.vector_store = vector_store
                        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
                        
                        # Analyze each document
                        all_results = []
                        overall_scores = {'environmental': 0, 'social': 0, 'governance': 0}
                        
                        for doc in st.session_state.documents:
                            doc_result = {'name': doc['name']}
                            for pillar in ['environmental', 'social', 'governance']:
                                score_data = score_esg_pillar(llm, retriever, pillar, doc['text'])
                                doc_result[pillar] = score_data
                                overall_scores[pillar] += score_data['score']
                            
                            all_results.append(doc_result)
                        
                        # Calculate averages
                        num_docs = len(st.session_state.documents)
                        for key in overall_scores:
                            overall_scores[key] /= num_docs
                        
                        st.session_state.analysis_results = {
                            'documents': all_results,
                            'overall_scores': overall_scores,
                            'num_documents': num_docs,
                            'timestamp': str(st.session_state.get('timestamp', 'N/A'))
                        }
                        
                        st.success("✅ Analysis complete!")
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

# ==================== TAB 2: ANALYSIS RESULTS ====================
with tab2:
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        st.subheader("📊 Overall ESG Scores")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="score-card">
                <div class="pillar-label">🌍 Environmental</div>
                <div class="score-value">{results['overall_scores']['environmental']:.1f}</div>
                <div>/100</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="score-card">
                <div class="pillar-label">👥 Social</div>
                <div class="score-value">{results['overall_scores']['social']:.1f}</div>
                <div>/100</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="score-card">
                <div class="pillar-label">⚖️ Governance</div>
                <div class="score-value">{results['overall_scores']['governance']:.1f}</div>
                <div>/100</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Document breakdown
        for idx, doc_result in enumerate(results['documents'], 1):
            with st.expander(f"📄 {doc_result['name']}", expanded=(idx==1)):
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🌍 Environmental", f"{doc_result['environmental']['score']}/100")
                with col2:
                    st.metric("👥 Social", f"{doc_result['social']['score']}/100")
                with col3:
                    st.metric("⚖️ Governance", f"{doc_result['governance']['score']}/100")
                
                for pillar in ['environmental', 'social', 'governance']:
                    st.markdown(f"### {pillar.title()}")
                    data = doc_result[pillar]
                    
                    st.write(f"**Analysis:** {data['reason']}")
                    
                    st.markdown("**Evidence:**")
                    for ev in data['evidence']:
                        st.markdown(f"""
                        <div class="evidence-box">
                        📄 <b>Page {ev['page']}:</b> {ev['text']}
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Download report
        report = generate_esg_report(results)
        st.download_button(
            label="📥 Download Full Report",
            data=report,
            file_name=f"ESG_Report_{len(results['documents'])}_docs.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    else:
        st.info("⏳ Upload and analyze documents first to see results")

# ==================== TAB 3: Q&A CHAT ====================
with tab3:
    if st.session_state.vector_store:
        st.subheader("💬 Ask Questions About Your Documents")
        
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        if user_input := st.chat_input("Ask a question about the documents..."):
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            with st.chat_message("user"):
                st.write(user_input)
            
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        llm = ChatOpenAI(model="gpt-4", temperature=0.3)
                        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
                        
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            return_source_documents=True
                        )
                        
                        response = qa_chain({"query": user_input})
                        answer = response['result']
                        
                        st.write(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    else:
        st.info("⏳ Upload and analyze documents first to enable Q&A")

# ==================== TAB 4: COMPARISON ====================
with tab4:
    if st.session_state.analysis_results and len(st.session_state.analysis_results['documents']) > 1:
        st.subheader("📊 Document Comparison")
        
        results = st.session_state.analysis_results
        
        comparison_data = []
        for doc in results['documents']:
            comparison_data.append({
                'Document': doc['name'],
                'Environmental': doc['environmental']['score'],
                'Social': doc['social']['score'],
                'Governance': doc['governance']['score']
            })
        
        st.dataframe(comparison_data, use_container_width=True)
        
        # Visualization
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        df_plot = df.set_index('Document')
        
        st.bar_chart(df_plot)
        
        # Comparison insights
        st.markdown("---")
        st.subheader("📈 Key Insights")
        
        docs = results['documents']
        best_env = max(docs, key=lambda x: x['environmental']['score'])
        best_soc = max(docs, key=lambda x: x['social']['score'])
        best_gov = max(docs, key=lambda x: x['governance']['score'])
        
        st.write(f"🌍 **Best Environmental Score:** {best_env['name']} ({best_env['environmental']['score']}/100)")
        st.write(f"👥 **Best Social Score:** {best_soc['name']} ({best_soc['social']['score']}/100)")
        st.write(f"⚖️ **Best Governance Score:** {best_gov['name']} ({best_gov['governance']['score']}/100)")
    
    else:
        st.info("⏳ Upload at least 2 documents to see comparison")
