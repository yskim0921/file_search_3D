# search_with_3d.py (기존 구조 + 3D 시각화)

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import TypedDict, List
from langchain import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
import pymysql

# 🔧 수정: 상태 정의에 벡터 정보 추가
class AgentState(TypedDict):
    query: str
    keywords: str
    file_paths: list
    relevant_docs: str
    result: str
    search_vectors: list  # 3D 시각화용 벡터 데이터 추가

# DB 접속 정보
DB_CONFIG = {
    'host': 'localhost',
    'user': 'admin',
    'password': '1qazZAQ!',
    'db': 'final',
    'charset': 'utf8mb4'
}

# PCA 함수
def robust_pca(X, n_components=3):
    try:
        X = np.array(X, dtype=np.float64)
        if X.shape[0] < 2:
            return X
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        n_components = min(n_components, X_centered.shape[1], X_centered.shape[0])
        V = Vt[:n_components].T
        X_reduced = X_centered @ V
        return np.real(X_reduced)
    except Exception as e:
        st.error(f"PCA 계산 오류: {str(e)}")
        return X[:, :min(3, X.shape[1])] if X.shape[1] >= 2 else np.column_stack([X.flatten(), np.zeros(X.shape[0]), np.zeros(X.shape[0])])

# Streamlit 설정
st.set_page_config(page_title="🔍 문서 검색 & 3D 시각화", layout="wide")
st.title("🔍 문서 검색 & 3D 시각화")

# 캐시된 리소스
@st.cache_resource
def get_llm():
    return Ollama(model="exaone3.5:2.4b")

@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model="exaone3.5:2.4b")

llm = get_llm()
embeddings = get_embeddings()

# 키워드 추출 에이전트 (기존과 동일)
keyword_prompt = PromptTemplate.from_template(
    """사용자의 질문에서 띄어쓰기 확인하고 찾고자 하는 키워드를 쉼표로 구분하여 출력하세요.
    벡터스토어 검색을 위한 최적의 키워드를 추출해주세요.
    \n질문: {query}"""
)

def extractor_agent(state: AgentState):
    chain = LLMChain(llm=llm, prompt=keyword_prompt)
    keywords = chain.run({"query": state["query"]})
    return {**state, "keywords": keywords.strip()}

# 🔧 수정: RAG + 파일 정보 + 벡터 정보 추출 에이전트
def rag_agent(state: AgentState):
    vectorstore = Chroma(
        persist_directory="./rag_chroma/documents/summary/", 
        embedding_function=embeddings
    )
    
    # 벡터스토어 검색 (점수 포함) - 기존과 동일
    results = vectorstore.similarity_search_with_score(state["keywords"], k=10)
    
    if not results:
        return {
            **state,
            "file_paths": [],
            "relevant_docs": "",
            "search_vectors": []
        }
    
    # 점수 범위 계산 - 기존과 동일
    scores = [score for _, score in results]
    min_score = min(scores)
    max_score = max(scores)
    
    # 🔧 추가: 쿼리 벡터 생성
    query_vector = embeddings.embed_query(state["keywords"])
    search_vectors = []
    
    # MySQL 연결 및 파일 정보 조회 - 기존과 동일
    conn = None
    file_entries = []
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        for doc, score in results:
            # 🔧 기존 유사도 계산 로직 그대로 사용
            if max_score == min_score:
                similarity = 100.0
            else:
                similarity = (1 - (score - min_score)/(max_score - min_score)) * 100
            
            # 문서 ID 추출
            doc_id = doc.metadata.get('id')
            
            # 🔧 추가: 문서 벡터 생성 (3D 시각화용)
            doc_text = doc.page_content
            try:
                doc_vector = embeddings.embed_query(doc_text[:1000])  # 텍스트 길이 제한
                search_vectors.append({
                    'vector': doc_vector,
                    'text': doc_text[:200],
                    'similarity': similarity,
                    'doc_id': doc_id,
                    'is_query': False
                })
            except:
                continue
            
            # MySQL에서 파일 정보 조회 - 기존과 동일
            if doc_id:
                cursor.execute("""
                    SELECT file_name, file_location, summary 
                    FROM documents 
                    WHERE id = %s
                """, (doc_id,))
                row = cursor.fetchone()
                if row:
                    # 🔧 추가: 벡터 데이터에 파일 정보 추가
                    if search_vectors:
                        search_vectors[-1]['file_name'] = row[0]
                        search_vectors[-1]['file_location'] = row[1]
                        search_vectors[-1]['full_summary'] = row[2]
                    
                    file_entries.append({
                        'name': row[0],
                        'location': row[1],
                        'summary': row[2],  # 전체 요약 저장
                        'relevance': round(similarity, 1)
                    })
    except Exception as e:
        st.error(f"DB 오류: {e}")
    finally:
        if conn: 
            conn.close()
    
    # 🔧 추가: 쿼리 벡터도 추가
    search_vectors.append({
        'vector': query_vector,
        'text': f"검색 쿼리: {state['keywords']}",
        'similarity': 100.0,
        'doc_id': 'QUERY',
        'is_query': True
    })
    
    # 문서 요약 생성 - 기존과 동일
    doc_summary = "\n".join([d.page_content for d, _ in results])
    
    return {
        **state, 
        "file_paths": file_entries, 
        "relevant_docs": doc_summary,
        "search_vectors": search_vectors  # 🔧 추가
    }

# 최종 답변 에이전트 - 기존과 동일
answer_prompt = PromptTemplate.from_template(
    """
    --- 제공된 정보 ---
    질문: {query}
    키워드: {keywords}
    문서 요약:
    {relevant_docs}
    파일 경로:
    {file_paths}
    
    ⚠️ 절대 지어내거나 추측하지 마세요.
    제공된 정보 안에 없으면 " 관련 정보가 없습니다"라고만 답변하세요.

    출력 규칙:
    각 파일에 대해 다음 형식으로 정확히 출력하세요:
    # 파일명: [파일명]
    # 파일위치: [파일 전체 경로]
    # 관련성: [소수점 1자리 %]
    # 설명: [문서 요약]

    관련 정보가 없으면 " 관련 정보가 없습니다"
    """
)

def answer_agent(state: AgentState):
    entries = state.get("file_paths", [])
    if not entries:
        return {**state, "result": " 관련 정보가 없습니다"}
    
    answer = ""
    for i, entry in enumerate(entries, 1):
        answer += f"{i}순위\n"
        answer += f"# 파일명: {entry['name']}\n"
        answer += f"# 파일위치: {entry['location']}\n"
        answer += f"# 관련성: {entry['relevance']}%\n"
        answer += f"# 설명: {entry['summary']}\n\n"  # 🔧 수정: 전체 요약 표시
    
    return {**state, "result": answer.strip()}

# LangGraph 연결 - 기존과 동일
graph = StateGraph(AgentState)
graph.add_node("extractor", extractor_agent)
graph.add_node("rag", rag_agent)
graph.add_node("answer", answer_agent)
graph.set_entry_point("extractor")
graph.add_edge("extractor", "rag")
graph.add_edge("rag", "answer")
graph.add_edge("answer", END)
app = graph.compile()

# 🔧 새로운 부분: Streamlit UI
st.sidebar.header("⚙️ 설정")
query = st.sidebar.text_input("검색어를 입력하세요:", value="에이닷비지")

if st.sidebar.button("🔍 검색 실행", type="primary"):
    if query:
        with st.spinner("검색 중..."):
            # 🔧 기존 LangGraph 파이프라인 실행
            state = {
                "query": query, 
                "keywords": "", 
                "file_paths": [], 
                "relevant_docs": "", 
                "result": "",
                "search_vectors": []  # 🔧 추가
            }
            result = app.invoke(state)
            
            # 세션 상태에 결과 저장
            st.session_state.search_result = result
    else:
        st.sidebar.error("검색어를 입력해주세요.")

# 결과 표시
if 'search_result' in st.session_state:
    result = st.session_state.search_result
    
    # 2열 레이아웃
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 검색 결과")
        
        # 🔧 기존 텍스트 결과 표시
        st.text_area(
            "검색 결과 (기존 형식)",
            value=result["result"],
            height=500,
            key="text_result"
        )
        
        # 추가 정보
        if result.get("file_paths"):
            st.subheader("📊 요약 정보")
            st.metric("검색된 문서 수", len(result["file_paths"]))
            if result["file_paths"]:
                avg_relevance = sum(item['relevance'] for item in result["file_paths"]) / len(result["file_paths"])
                st.metric("평균 관련성", f"{avg_relevance:.1f}%")
    
    with col2:
        st.subheader("🌌 3D 시각화")
        
        # 🔧 3D 시각화
        if result.get("search_vectors") and len(result["search_vectors"]) > 1:
            try:
                # 벡터 데이터 준비
                valid_vectors = []
                valid_items = []
                
                for item in result["search_vectors"]:
                    if 'vector' in item and isinstance(item['vector'], list):
                        try:
                            vector_array = np.array(item['vector'], dtype=np.float64)
                            if not np.isnan(vector_array).any():
                                valid_vectors.append(vector_array)
                                valid_items.append(item)
                        except:
                            continue
                
                if len(valid_vectors) >= 2:
                    vectors = np.array(valid_vectors)
                    vectors_3d = robust_pca(vectors, n_components=3)
                    
                    # 시각화 데이터 준비
                    colors = []
                    sizes = []
                    texts = []
                    symbols = []
                    
                    for item in valid_items:
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
                            sizes.append(max(8, 10 + similarity/10))
                            symbols.append('circle')
                            
                            file_name = item.get('file_name', 'Unknown')
                            summary = item.get('full_summary', item['text'])[:300]
                            texts.append(f"{file_name}<br>유사도: {similarity:.1f}%<br>{summary}")
                    
                    # Plotly 3D 그래프
                    fig = go.Figure()
                    
                    # 점들 추가
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
                                line=dict(width=2, color='DarkSlateGrey'),
                                opacity=0.9 if valid_items[i]['is_query'] else 0.7
                            ),
                            text=texts[i],
                            hoverinfo='text',
                            showlegend=False,
                            name=f"Document {i}"
                        ))
                    
                    # 연결선 추가
                    query_idx = len(vectors_3d) - 1
                    for i in range(len(vectors_3d) - 1):
                        fig.add_trace(go.Scatter3d(
                            x=[float(vectors_3d[query_idx, 0]), float(vectors_3d[i, 0])],
                            y=[float(vectors_3d[query_idx, 1]), float(vectors_3d[i, 1])],
                            z=[float(vectors_3d[query_idx, 2]), float(vectors_3d[i, 2])],
                            mode='lines',
                            line=dict(color='lightgray', width=2, dash='dot'),
                            opacity=0.4,
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    
                    fig.update_layout(
                        title="검색 결과의 3D 벡터 공간",
                        scene=dict(
                            xaxis_title="주제 유사성",
                            yaxis_title="의미적 복잡도",
                            zaxis_title="문맥 관련성",
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                            aspectmode='cube'
                        ),
                        height=500,
                        showlegend=False,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 범례
                    st.markdown("""
                    **시각화 설명:**
                    - 🔶 빨간 다이아몬드: 검색 쿼리
                    - 🟢 초록색 점: 높은 유사도 문서  
                    - 🔵 파란색 점: 낮은 유사도 문서
                    - 점선: 쿼리와 문서 간 연결
                    """)
                else:
                    st.warning("시각화할 벡터 데이터가 부족합니다.")
                    
            except Exception as e:
                st.error(f"3D 시각화 오류: {str(e)}")
        else:
            st.info("검색을 실행하면 3D 시각화가 표시됩니다.")

# 🔧 콘솔 출력도 유지 (기존 방식)
if st.sidebar.button("🖥️ 콘솔 실행 (기존 방식)"):
    query = "에이닷비지"
    state = {"query": query, "keywords": "", "file_paths": [], "relevant_docs": "", "result": "", "search_vectors": []}
    result = app.invoke(state)
    st.sidebar.code(result["result"])

# 초기화 버튼
if st.sidebar.button("🔄 초기화"):
    if 'search_result' in st.session_state:
        del st.session_state.search_result
    st.rerun()