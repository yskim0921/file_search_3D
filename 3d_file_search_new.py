from typing import TypedDict, List, Dict, Any
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langgraph.graph import StateGraph, END # LangGraph 컴포넌트는 여전히 로직 제어

import plotly.graph_objects as go
import plotly.offline as pyo  # 추가: HTML 파일 저장 및 브라우저 열기용
import numpy as np
import ipywidgets as widgets  # .py 스크립트에서는 사용 안 함 (클래스 내부 유지, 하지만 show_visualization에서 무시)
from IPython.display import display  # .py 스크립트에서는 사용 안 함
import pymysql  # MySQL 연결용
import os  # 추가: 디렉토리 생성용
import re  # 추가: 파일명 안전 처리용

# ================================================================
# 1. 상태 정의 (AgentState)
# ================================================================
class AgentState(TypedDict):
    query: str
    keywords: str
    search_results: List[Dict[str, Any]]
    context: str
    result: str

# ================================================================
# 2. 데이터베이스 접속 정보
# ================================================================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'admin',
    'password': '1qazZAQ!',
    'db': 'final',
    'charset': 'utf8mb4'
}

# ================================================================
# 3. ChromaDB 및 LLM 설정
# ================================================================
CHROMA_PATH = "./rag_chroma/documents/title_summary_test/"
EMBEDDINGS = OllamaEmbeddings(model="exaone3.5:2.4b")
LLM = Ollama(model="exaone3.5:2.4b")
CHAT_LLM = ChatOllama(model="exaone3.5:2.4b", temperature=0.1)

# ================================================================
# 4. 노트북용 3D 시각화 클래스 (RAGNotebookVisualizer) - .py 스크립트 호환으로 수정
# - Jupyter용 display/ipywidgets 제거, HTML 파일 저장으로 전환
# ================================================================
class RAGNotebookVisualizer:
    def __init__(self):
        # 질문(쿼리) 노드의 3D 공간 위치를 중앙으로 고정
        self.query_position = (0.0, 0.0, 0.0)
        self.current_query = None  # 추가: 현재 쿼리 저장 (파일명 생성용)

        self.fig3d = go.Figure()  # FigureWidget -> Figure (스크립트용)
        self._init_scene()  # 3D 장면 초기화

        # 관련성 막대 그래프 (검색 결과)
        self.bar_fig = go.Figure()  # FigureWidget -> Figure
        self.bar_fig.add_trace(go.Bar(x=[], y=[], text=[], textposition='auto'))
        self.bar_fig.update_layout(
            title='검색 결과 관련성 (%)',
            height=260,
            margin=dict(t=40)
        )

        # Jupyter용 컨테이너 제거 (스크립트에서는 사용 안 함)

    def _init_scene(self):
        """3D 장면 초기화: 질문 중심 배치"""
        # 1. 쿼리 노드 (중앙, 빨간색 다이아몬드)
        self.query_trace_idx = 0 # 가장 먼저 추가되는 trace
        self.fig3d.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], # 중앙 고정
            mode='markers+text',
            marker=dict(
                size=5, # 이전보다 약간 크게
                color='red',
                symbol='diamond',
                line=dict(width=2, color='darkred')
            ),
            text=['❓ Query'], # 초기 텍스트
            textposition='bottom center',
            name='사용자 질문',
            showlegend=True
        ))
        
        # 2. 검색 결과 노드 (초기 empty, update_search_results에서 채움)
        self.search_trace_idx = 1
        self.fig3d.add_trace(go.Scatter3d(
            x=[], y=[], z=[],
            mode='markers+text',
            marker=dict(
                size=[], 
                color=[], 
                colorscale='Viridis', # 관련성 높을수록 밝은 노란색
                cmin=0, cmax=100, 
                opacity=0.9, 
                showscale=True,
                colorbar=dict(title='관련성 (%)')
            ),
            text=[], 
            textposition='top center',
            name='검색 결과'
        ))
        
        # 3. 쿼리-결과 연결선 (빨간 점선)
        self.query_edge_trace_idx = 2
        self.fig3d.add_trace(go.Scatter3d(
            x=[], y=[], z=[],
            mode='lines',
            line=dict(color='rgba(255,0,0,0.7)', width=2, dash='dot'),
            name='쿼리-문서 연결',
            showlegend=False # 범례에 나타내지 않음
        ))
        
        # 4. 레이아웃 설정 (X, Y 축 숨김, Z 축은 거리 의미)
        self.fig3d.update_layout(
            title='🔍 RAG: 질문 기반 문서 검색 (가까울수록 관련성 높음)',
            scene=dict(
                xaxis=dict(visible=False, showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(visible=False, showticklabels=False, showgrid=False, zeroline=False),
                zaxis=dict(title='거리 (관련성 ↓)', showgrid=True), # Z축이 거리를 나타냄
                camera=dict(eye=dict(x=1.1, y=1.1, z=0.8)) # 초기 카메라 시점 조정
            ),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

    def show_visualization(self):
        """스크립트 실행 후 브라우저에 그래프를 띄웁니다."""
        if self.current_query is None:
            print("[ERROR] 현재 쿼리가 설정되지 않았습니다. 검색을 먼저 실행하세요.")
            return
        
        # 파일명 생성: 쿼리 내용을 기반으로 파일명 생성
        # 특수문자 제거 및 공백을 언더스코어로 변경하여 안전한 파일명 생성
        safe_query = re.sub(r'[\\/*?:"<>|]', "", self.current_query) # 윈도우 파일명 금지 문자 제거
        safe_query = safe_query.strip().replace(" ", "_")
        if len(safe_query) > 50: # 파일명이 너무 길어지지 않도록 제한
            safe_query = safe_query[:50]
            
        # 저장 폴더 설정
        output_dir = "search"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        filename = os.path.join(output_dir, f"{safe_query}_3d_visualization.html")
        
        # 3D 그래프 저장 및 브라우저 열기
        print(f"\n[INFO] 3D 시각화 결과를 '{filename}' 파일로 저장하고 웹 브라우저에서 표시합니다.")
        pyo.plot(self.fig3d, filename=filename, auto_open=True)
        
        # 막대 그래프 저장 및 브라우저 열기
        bar_filename = os.path.join(output_dir, f"{safe_query}_bar_chart.html")
        print(f"[INFO] 막대 그래프를 '{bar_filename}' 파일로 저장하고 웹 브라우저에서 표시합니다.")
        pyo.plot(self.bar_fig, filename=bar_filename, auto_open=True)

    def _update_query_text(self, query: str):
        """중앙 쿼리 노드의 텍스트 업데이트"""
        q_text = f"❓ {query[:30]}" + ("..." if len(query) > 30 else "")
        self.fig3d.data[self.query_trace_idx].text = [q_text]
        self.current_query = query  # 추가: 현재 쿼리 저장

    def update_search_results(self, results: List[Dict[str, Any]], current_query: str):
        """검색 결과를 3D 그래프와 막대 차트에 업데이트"""
        if not results: return # 결과 없으면 업데이트 안 함
        
        # 쿼리 노드 텍스트를 먼저 업데이트 (current_query 저장 포함)
        self._update_query_text(current_query)

        # 1. 관련성 순으로 내림차순 정렬
        res_sorted = sorted(results, key=lambda x: x.get("relevance", 0), reverse=True)
        n = len(res_sorted)
        
        bx, by, bz = self.query_position # 쿼리 노드 중앙 좌표

        xs, ys, zs = [], [], []
        edge_xs, edge_ys, edge_zs = [], [], []
        sizes, colors, texts = [], [], []

        # 배치 파라미터: 유사도가 높을수록 쿼리에 가까움
        min_dist = 0.5  # 관련성 100%일 때의 최소 거리
        max_dist = 4.0  # 관련성 0%일 때의 최대 거리
        
        # Golden angle spiral 배치로 구형 표면에 고르게 분포 (theta용)
        golden_angle = np.pi * (3 - np.sqrt(5)) # 약 2.39996 라디안

        for i, item in enumerate(res_sorted):
            rel = float(item.get("relevance", 0.0))
            
            # 거리를 관련성에 반비례하게 설정 (rel 높으면 dist 작아짐)
            dist = min_dist + (1 - rel / 100) * (max_dist - min_dist)
            
            # 구면 좌표 계산
            theta = i * golden_angle # 0 ~ 2pi (방위각: 고르게 분포)
            
            # phi 계산을 치우치게 변경 (관련성 높은 문서들이 z축 positive 방향으로 클러스터링되도록 기울기 skew)
            u = (i + 0.5) / max(n, 1)
            skew_power = 2.0  # 조정 가능: >1 치우침 강도 증가 (e.g., 3.0 더 강하게 클러스터)
            phi = np.pi * (u ** skew_power)  # skewed phi: 고 rel -> 작은 phi (z positive 치우침)
            
            x = bx + dist * np.sin(phi) * np.cos(theta)
            y = by + dist * np.sin(phi) * np.sin(theta)
            z = bz + dist * np.cos(phi) # z: 관련성 낮을수록 (phi 커짐) cos 작아지거나 음수 가능
            
            xs.append(x); ys.append(y); zs.append(z)
            
            # 쿼리(중앙)에서 각 문서 노드로 연결선 추가
            edge_xs.extend([bx, x, None])
            edge_ys.extend([by, y, None])
            edge_zs.extend([bz, z, None])
            
            sizes.append(max(8, rel / 4.0)) # 관련성 높으면 마커 크기 크게
            colors.append(rel) # 색상
            texts.append(f"{item.get('file_name','문서')}<br>{rel}%")

        # 3D 그래프의 trace 업데이트 (batch_update 대신 직접 업데이트 - Figure용)
        self.fig3d.data[self.search_trace_idx].x = xs
        self.fig3d.data[self.search_trace_idx].y = ys
        self.fig3d.data[self.search_trace_idx].z = zs
        self.fig3d.data[self.search_trace_idx].marker.size = sizes
        self.fig3d.data[self.search_trace_idx].marker.color = colors
        self.fig3d.data[self.search_trace_idx].text = texts
        
        self.fig3d.data[self.query_edge_trace_idx].x = edge_xs
        self.fig3d.data[self.query_edge_trace_idx].y = edge_ys
        self.fig3d.data[self.query_edge_trace_idx].z = edge_zs
        
        # 막대 차트 업데이트
        file_names_bar = [res['file_name'] for res in res_sorted]
        relevances_bar = [res['relevance'] for res in res_sorted]
        self.bar_fig.data[0].x = file_names_bar
        self.bar_fig.data[0].y = relevances_bar
        self.bar_fig.data[0].text = [f"{r}%" for r in relevances_bar]
        self.bar_fig.data[0].marker.color = relevances_bar
        self.bar_fig.data[0].marker.colorscale = 'Viridis'
        self.bar_fig.data[0].marker.cmin = 0
        self.bar_fig.data[0].marker.cmax = 100
        self.bar_fig.update_layout(
            xaxis_tickangle=-45,
            margin=dict(b=100, t=40)
        )

# ================================================================
# 5. 시각화 객체 생성
# ================================================================
visualizer = RAGNotebookVisualizer()

# ================================================================
# 6. RAG 에이전트 함수 (시각화 로직에서 에이전트 프로세스 제거)
# ================================================================
def extractor_agent(state: AgentState):
    """사용자 쿼리에서 키워드 추출"""
    # 쿼리가 입력되면 3D 시각화 중앙의 텍스트를 업데이트
    visualizer._update_query_text(state["query"])
    
    keyword_prompt = PromptTemplate.from_template(
        """사용자의 질문에서 띄어쓰기 확인하고 찾고자 하는 키워드를 쉼표로 구분하여 출력하세요.
        벡터스토어 검색을 위한 최적의 키워드를 추출해주세요.
        \n질문: {query}"""
    )
    formatted_prompt = keyword_prompt.format(query=state["query"])
    keywords = LLM.invoke(formatted_prompt)
    
    return {**state, "keywords": keywords.strip()}

def rag_search_agent(state: AgentState):
    """검색 및 관련성 계산, 3D 시각화 업데이트"""
    try:
        # 1. ChromaDB 로드 및 검색
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDINGS)
        results = vectorstore.similarity_search_with_score(state["keywords"], k=10)

        if not results: 
            # 검색 결과가 없으면 3D 시각화에서 문서 노드를 비움
            visualizer.update_search_results([], state["query"])
            return {**state, "search_results": [], "context": ""}
        
        # 2. 문서 중복 제거 (가장 높은 유사도 유지)
        best_doc_info = {}
        for doc, score in results:
            doc_id = doc.metadata.get("id")
            if doc_id:
                content = doc.page_content.strip()
                if content:
                    if doc_id not in best_doc_info or score < best_doc_info[doc_id][0]:
                        best_doc_info[doc_id] = (score, content)
        
        if not best_doc_info: 
            visualizer.update_search_results([], state["query"])
            return {**state, "search_results": [], "context": ""}
        
        # 3. 유사도 정규화 (0-100% 스케일)
        scores = [score for score, _ in best_doc_info.values()]
        min_score, max_score = min(scores), max(scores)
        
        # 4. MySQL에서 파일 메타데이터 조회
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        search_results_with_metadata = []
        
        for doc_id, (score, content) in best_doc_info.items():
            cursor.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
            row = cursor.fetchone()
            if row:
                if max_score != min_score:
                    relevance = (1 - (score - min_score) / (max_score - min_score)) * 98
                else:
                    relevance = 98.0
                search_results_with_metadata.append({
                    "relevance": round(relevance, 1),
                    "file_name": row["file_name"],
                    "file_location": row["file_location"],
                    "summary": row["summary"][:100] + "..." if row["summary"] else "요약 없음",
                    "doc_type": row["doc_type"],
                    "keywords": row["keywords"],
                    "content": content
                })
        conn.close()
        
        # 5. 관련성 순으로 정렬 및 컨텍스트 생성
        search_results_with_metadata.sort(key=lambda x: x["relevance"], reverse=True)
        context = "\n\n".join([res['content'] for res in search_results_with_metadata[:10]])
        
        # 6. 3D 시각화 업데이트
        visualizer.update_search_results(search_results_with_metadata, state["query"])
        
        return {**state, "search_results": search_results_with_metadata, "context": context}
    
    except Exception as e:
        print(f"❌ RAG 검색 오류: {e}")
        visualizer.update_search_results([], state["query"]) 
        return {**state, "search_results": [], "context": ""}

def answer_generator_agent(state: AgentState):
    """검색된 문서를 바탕으로 답변 생성"""
    if not state["search_results"]:
        return {**state, "result": "관련 정보를 찾을 수 없습니다."}
    
    search_summary = "\n".join([
        f"- {res['file_name']} ({res['relevance']}%)" 
        for res in state["search_results"]
    ])
    
    prompt = ChatPromptTemplate.from_template(
        """다음 검색 결과 요약과 문서 내용을 바탕으로 사용자 질문에 답변해 주세요.
        - 문서에 직접 언급된 내용만을 바탕으로 답변해야 합니다.
        - 정보가 없는 경우 "정보가 없습니다"라고 명확히 답변해 주세요.
        - 항상 공손하고 전문적인 어조를 유지해 주세요.

        ### 검색 결과 요약:
        {search_summary}

        ### 문서 내용:
        {context}

        ### 사용자 질문:
        {query}

        ### 답변:
        ==최종결론==
        찾은 문서중에 가장 관련이 높은 파일을 찾아주고 자세한 파일내용 설명해줘
        (순위, 무슨파일인지, 어디에 있는지, 요약은 무엇인지, 질문에 포함되는 문서키워드 등)
        """
    )
    chain = prompt | CHAT_LLM | StrOutputParser()
    result = chain.invoke({
        "search_summary": search_summary,
        "context": state["context"],
        "query": state["query"]
    })
    
    return {**state, "result": result.strip()}

def result_formatter_agent(state: AgentState):
    """최종 답변 및 검색 결과 형식화"""
    formatted_result = "🔍 검색된 문서 목록:\n"
    if not state["search_results"]:
        formatted_result += "  - 검색 결과 없음\n"
    else:
        for i, res in enumerate(state["search_results"], 1):
            formatted_result += (
                f"\n--- {i}순위 ({res['relevance']}%) ---\n"
                f"📄 파일명: {res['file_name']}\n"
                f"   📁 위치: {res['file_location']}\n"
                f"   📝 요약: {res['summary']}\n"
                f"   🗝️ 키워드: {res['keywords']}\n"
                f"   🏷️ 유형: {res['doc_type']}\n"
            )
    
    formatted_result += f"\n💬 AI 답변:\n{state['result']}"
    return {**state, "result": formatted_result}

# ================================================================
# 7. LangGraph 구성 및 실행
# ================================================================
graph = StateGraph(AgentState)
graph.add_node("extractor", extractor_agent)
graph.add_node("rag_search", rag_search_agent)
graph.add_node("answer_generator", answer_generator_agent)
graph.add_node("result_formatter", result_formatter_agent)

graph.set_entry_point("extractor")
graph.add_edge("extractor", "rag_search")
graph.add_edge("rag_search", "answer_generator")
graph.add_edge("answer_generator", "result_formatter")
graph.add_edge("result_formatter", END)

app = graph.compile()

# ================================================================
# 8. 실행 예시 (.py 스크립트용)
# ================================================================
# RAG 파이프라인 실행
state = {"query": "1111카카오톡", "keywords": "", "search_results": [], "context": "", "result": ""}
result = app.invoke(state)

print("\n" + "="*50)
print("📊 RAG 처리 완료! 최종 결과:")
print("="*50)
print(result["result"])

# 시각화 HTML 파일 생성 및 브라우저 열기
visualizer.show_visualization()