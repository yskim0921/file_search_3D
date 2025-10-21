# search_visualize_app.py (요약 내용 전체 표시 버전)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import TypedDict, List
from langchain import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langgraph.graph import StateGraph, END
import pymysql

# 기존 코드의 설정들
class AgentState(TypedDict):
    query: str
    keywords: str
    file_paths: list
    relevant_docs: str
    result: str
    search_vectors: list

DB_CONFIG = {
    'host': 'localhost',
    'user': 'admin',
    'password': '1qazZAQ!',
    'db': 'final',
    'charset': 'utf8mb4'
}

# SVD 기반 PCA (이전과 동일)
def robust_pca(X, n_components=3):
    try:
        X = np.array(X, dtype=np.float64)
        
        if X.shape[0] < 2:
            st.warning("데이터가 너무 적어 PCA를 수행할 수 없습니다.")
            return X
        
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        n_components = min(n_components, X_centered.shape[1], X_centered.shape[0])
        V = Vt[:n_components].T
        X_reduced = X_centered @ V
        X_reduced = np.real(X_reduced)
        
        return X_reduced
        
    except Exception as e:
        st.error(f"PCA 계산 오류: {str(e)}")
        if X.shape[1] >= 2:
            return X[:, :min(3, X.shape[1])]
        else:
            return np.column_stack([X.flatten(), np.zeros(X.shape[0]), np.zeros(X.shape[0])])

# Streamlit 설정
st.set_page_config(page_title="🔍 3D 검색 시각화", layout="wide")
st.title("🔍 문서 검색 & 3D 시각화")
st.markdown("검색 결과를 3D 공간에서 시각화하여 문서 간의 관계를 탐색합니다.")

# 세션 상태 초기화
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'vectors_data' not in st.session_state:
    st.session_state.vectors_data = None

@st.cache_resource
def get_llm():
    return Ollama(model="exaone3.5:2.4b")

@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model="exaone3.5:2.4b")

llm = get_llm()
embeddings = get_embeddings()

def rag_agent_with_vectors(state: AgentState):
    """벡터스토어에서 검색하고 벡터 정보도 함께 반환"""
    try:
        vectorstore = Chroma(
            persist_directory="./rag_chroma/documents/summary/", 
            embedding_function=embeddings
        )
        
        results = vectorstore.similarity_search_with_score(state["keywords"], k=10)
        
        if not results:
            return {
                **state,
                "file_paths": [],
                "relevant_docs": "",
                "search_vectors": []
            }
        
        search_vectors = []
        file_entries = []
        
        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)
        
        query_vector = embeddings.embed_query(state["keywords"])
        
        if not isinstance(query_vector, list) or len(query_vector) == 0:
            st.error("쿼리 벡터 생성에 실패했습니다.")
            return {**state, "file_paths": [], "relevant_docs": "", "search_vectors": []}
        
        # 🔧 수정: 문서 텍스트 길이 제한 제거
        for doc, score in results:
            doc_text = doc.page_content
            
            if not doc_text or len(doc_text.strip()) == 0:
                continue
                
            try:
                # 🔧 수정: 텍스트 길이 제한을 늘림 (500 → 2000)
                doc_vector = embeddings.embed_query(doc_text[:2000])
                
                if not isinstance(doc_vector, list) or len(doc_vector) != len(query_vector):
                    continue
                
                if max_score == min_score:
                    similarity = 100.0
                else:
                    similarity = (1 - (score - min_score)/(max_score - min_score)) * 100
                
                search_vectors.append({
                    'vector': doc_vector,
                    'text': doc_text,  # 🔧 수정: 전체 텍스트 저장 (자르지 않음)
                    'similarity': similarity,
                    'doc_id': doc.metadata.get('id'),
                    'is_query': False
                })
            except Exception as e:
                st.warning(f"문서 벡터 생성 실패: {str(e)}")
                continue
        
        search_vectors.append({
            'vector': query_vector,
            'text': f"검색 쿼리: {state['keywords']}",
            'similarity': 100.0,
            'doc_id': 'QUERY',
            'is_query': True
        })
        
        # MySQL에서 파일 정보 조회
        conn = None
        try:
            conn = pymysql.connect(**DB_CONFIG)
            cursor = conn.cursor()
            
            for item in search_vectors[:-1]:
                if item['doc_id']:
                    cursor.execute("""
                        SELECT file_name, file_location, summary 
                        FROM documents WHERE id = %s
                    """, (item['doc_id'],))
                    row = cursor.fetchone()
                    if row:
                        item['file_name'] = row[0]
                        item['file_location'] = row[1]
                        # 🔧 수정: 전체 요약 저장 (자르지 않음)
                        item['full_summary'] = row[2]
                        file_entries.append({
                            'name': row[0],
                            'location': row[1],
                            'summary': row[2],  # 🔧 수정: 전체 요약 저장
                            'relevance': round(item['similarity'], 1)
                        })
        except Exception as e:
            st.warning(f"DB 조회 오류: {e}")
        finally:
            if conn:
                conn.close()
        
        doc_summary = "\n".join([d.page_content for d, _ in results])
        
        return {
            **state, 
            "file_paths": file_entries, 
            "relevant_docs": doc_summary,
            "search_vectors": search_vectors
        }
        
    except Exception as e:
        st.error(f"검색 중 오류 발생: {str(e)}")
        return {**state, "file_paths": [], "relevant_docs": "", "search_vectors": []}

# 메인 UI
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔍 검색")
    query = st.text_input("검색어를 입력하세요:", placeholder="예: 카카오톡")
    
    if st.button("검색 실행", type="primary"):
        if query:
            with st.spinner("검색 중..."):
                try:
                    keyword_prompt = PromptTemplate.from_template(
                        """사용자의 질문에서 벡터스토어 검색을 위한 최적의 키워드를 추출하세요.
                        질문: {query}"""
                    )
                    chain = LLMChain(llm=llm, prompt=keyword_prompt)
                    keywords = chain.run({"query": query}).strip()
                    
                    st.info(f"추출된 키워드: {keywords}")
                    
                    state = {
                        "query": query,
                        "keywords": keywords,
                        "file_paths": [],
                        "relevant_docs": "",
                        "result": "",
                        "search_vectors": []
                    }
                    
                    result = rag_agent_with_vectors(state)
                    st.session_state.search_results = result['file_paths']
                    st.session_state.vectors_data = result['search_vectors']
                    
                    if not st.session_state.search_results:
                        st.warning("검색 결과가 없습니다. 다른 키워드를 시도해보세요.")
                    
                except Exception as e:
                    st.error(f"검색 오류: {str(e)}")
                    
        else:
            st.error("검색어를 입력해주세요.")
    
    # 🔧 수정: 검색 결과 표시 (요약 내용 전체 표시)
    if st.session_state.search_results:
        st.subheader("📄 검색 결과")
        
        # 전체 요약 보기 옵션 추가
        show_full_summary = st.checkbox("📖 전체 요약 보기", value=True)
        
        for i, entry in enumerate(st.session_state.search_results[:5], 1):
            with st.expander(f"{i}. {entry['name']} ({entry['relevance']}%)", expanded=False):
                st.markdown(f"**📁 파일 위치:** `{entry['location']}`")
                st.markdown(f"**🎯 관련성:** {entry['relevance']}%")
                
                # 🔧 수정: 요약 내용 전체 표시
                st.markdown("**📝 요약 내용:**")
                if show_full_summary:
                    # 전체 요약을 스크롤 가능한 텍스트 영역에 표시
                    st.text_area(
                        label="",
                        value=entry['summary'],
                        height=200,
                        key=f"summary_{i}",
                        label_visibility="collapsed"
                    )
                else:
                    # 간단 미리보기 (처음 300자)
                    preview = entry['summary'][:300]
                    if len(entry['summary']) > 300:
                        preview += "..."
                    st.info(preview)
                    
                    # 전체 보기 버튼
                    if st.button(f"전체 요약 보기", key=f"show_full_{i}"):
                        st.text_area(
                            label="전체 요약",
                            value=entry['summary'],
                            height=300,
                            key=f"full_summary_{i}"
                        )

with col2:
    st.subheader("🌌 3D 시각화")
    
    if st.session_state.vectors_data and len(st.session_state.vectors_data) > 1:
        try:
            valid_vectors = []
            valid_items = []
            
            for item in st.session_state.vectors_data:
                if 'vector' in item and isinstance(item['vector'], list) and len(item['vector']) > 0:
                    try:
                        vector_array = np.array(item['vector'], dtype=np.float64)
                        if not np.isnan(vector_array).any() and not np.isinf(vector_array).any():
                            valid_vectors.append(vector_array)
                            valid_items.append(item)
                    except:
                        continue
            
            if len(valid_vectors) < 2:
                st.warning("유효한 벡터 데이터가 부족합니다.")
            else:
                vectors = np.array(valid_vectors)
                
                with st.spinner("3D 변환 중..."):
                    vectors_3d = robust_pca(vectors, n_components=3)
                
                if vectors_3d is None or vectors_3d.shape[1] < 3:
                    st.error("3D 변환에 실패했습니다.")
                else:
                    colors = []
                    sizes = []
                    texts = []
                    symbols = []
                    
                    for i, item in enumerate(valid_items):
                        if item['is_query']:
                            colors.append('red')
                            sizes.append(20)
                            symbols.append('diamond')
                            texts.append(f"🔍 {item['text']}")
                        else:
                            similarity = item['similarity']
                            green_val = max(0, min(255, int(similarity * 2.55)))
                            red_val = max(0, min(255, 255 - green_val))
                            colors.append(f'rgb({red_val}, {green_val}, 100)')
                            sizes.append(max(5, 10 + similarity/10))
                            symbols.append('circle')
                            file_name = item.get('file_name', 'Unknown')
                            # 🔧 수정: 호버 텍스트도 더 자세히 표시
                            hover_text = f"{file_name}<br>유사도: {similarity:.1f}%<br>"
                            # 요약의 처음 500자만 호버에 표시 (너무 길면 화면이 복잡해짐)
                            summary_preview = item.get('full_summary', item['text'])[:500]
                            if len(item.get('full_summary', item['text'])) > 500:
                                summary_preview += "..."
                            hover_text += summary_preview
                            texts.append(hover_text)
                    
                    # Plotly 3D 산점도
                    fig = go.Figure()
                    
                    for i in range(len(vectors_3d)):
                        fig.add_trace(go.Scatter3d(
                            x=[float(vectors_3d[i, 0])],
                            y=[float(vectors_3d[i, 1])],
                            z=[float(vectors_3d[i, 2])],
                            mode='markers',
                            marker=dict(
                                size=sizes[i],
                                color=colors[i],
                                symbol=symbols[i],
                                line=dict(
                                    width=3 if valid_items[i]['is_query'] else 1,
                                    color='DarkSlateGrey'
                                ),
                                opacity=0.9 if valid_items[i]['is_query'] else 0.7
                            ),
                            text=texts[i],
                            hoverinfo='text',
                            showlegend=False,
                            name=f"Point {i}"
                        ))
                    
                    # 연결선 추가
                    query_idx = len(vectors_3d) - 1
                    for i in range(len(vectors_3d) - 1):
                        fig.add_trace(go.Scatter3d(
                            x=[float(vectors_3d[query_idx, 0]), float(vectors_3d[i, 0])],
                            y=[float(vectors_3d[query_idx, 1]), float(vectors_3d[i, 1])],
                            z=[float(vectors_3d[query_idx, 2]), float(vectors_3d[i, 2])],
                            mode='lines',
                            line=dict(
                                color='lightgray',
                                width=2,
                                dash='dot'
                            ),
                            opacity=0.4,
                            showlegend=False,
                            hoverinfo='skip',
                            name=f"Connection {i}"
                        ))
                    
                    fig.update_layout(
                        title="검색 결과의 3D 벡터 공간",
                        scene=dict(
                            xaxis_title="주제 유사성 (Topic Similarity)",
                            yaxis_title="의미적 복잡도 (Semantic Complexity)", 
                            zaxis_title="문맥 관련성 (Contextual Relevance)",
                            camera=dict(
                                eye=dict(x=1.5, y=1.5, z=1.5)
                            ),
                            aspectmode='cube'
                        ),
                        height=600,
                        showlegend=False,
                        margin=dict(l=0, r=0, t=50, b=0)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 통계 정보
                    if st.session_state.search_results:
                        col2_1, col2_2, col2_3 = st.columns(3)
                        with col2_1:
                            st.metric("검색된 문서", len(st.session_state.search_results))
                        with col2_2:
                            avg_sim = np.mean([item['relevance'] for item in st.session_state.search_results])
                            st.metric("평균 유사도", f"{avg_sim:.1f}%")
                        with col2_3:
                            max_sim = max([item['relevance'] for item in st.session_state.search_results])
                            st.metric("최고 유사도", f"{max_sim:.1f}%")
                
        except Exception as e:
            st.error(f"시각화 오류: {str(e)}")
            st.info("다시 검색을 시도해보세요.")
    else:
        st.info("검색을 실행하면 결과가 3D로 시각화됩니다.")

# 사이드바
with st.sidebar:
    st.header("⚙️ 시스템 상태")
    if st.session_state.vectors_data:
        st.success(f"벡터 {len(st.session_state.vectors_data)}개 로드됨")
    else:
        st.info("검색 대기 중")
    
    st.header("📋 도움말")
    st.markdown("""
    ### 사용법
    1. 검색어 입력 후 '검색 실행' 클릭
    2. 왼쪽에서 검색 결과 확인
    3. 오른쪽에서 3D 시각화 탐색
    
    ### 팁
    - expander를 클릭하면 전체 요약 내용 확인 가능
    - '전체 요약 보기' 체크박스로 표시 방식 변경
    - 3D 그래프는 마우스로 회전/확대 가능
    """)
    
    if st.button("🔄 초기화"):
        st.session_state.search_results = None
        st.session_state.vectors_data = None
        st.rerun()