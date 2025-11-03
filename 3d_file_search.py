# ================================================================
# 노트북용 RAG + 3D 시각화 통합 코드 (Python Script 버전)
# ================================================================
# - 질문(Query)을 중앙(0,0,0)에 고정
# - 유사도(관련성)가 높을수록 중앙에 가깝게 배치
# - Python 스크립트 실행을 위해 Plotly offline.plot() 방식 사용
# ================================================================

from typing import TypedDict, List, Dict, Any
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langgraph.graph import StateGraph, END
import pymysql
import sys
import os
import re # 파일명 정제를 위해 re 모듈 추가

# 시각화 관련
import plotly.graph_objects as go
import plotly.offline as pyo # 이 줄을 추가합니다.
import numpy as np

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
# 2. 데이터베이스 접속 정보 (환경에 맞게 수정하세요)
# ================================================================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'admin',
    'password': '1qazZAQ!',
    'db': 'final',
    'charset': 'utf8mb4'
}

# ================================================================
# 3. ChromaDB 및 LLM 설정 (환경에 맞게 수정하세요)
# ================================================================
CHROMA_PATH = "./rag_chroma/documents/title_summary_test/"
EMBEDDINGS = OllamaEmbeddings(model="exaone3.5:2.4b")
LLM = Ollama(model="exaone3.5:2.4b")
CHAT_LLM = ChatOllama(model="exaone3.5:2.4b", temperature=0.1)

# ================================================================
# 4. 3D 시각화 클래스 (Python Script용)
# ================================================================
class RAGVisualizerScript:
    def __init__(self):
        self.query_position = (0.0, 0.0, 0.0)
        self.fig3d = go.Figure() # FigureWidget 대신 일반 Figure 사용
        self._init_scene()
        self.current_query_for_filename = "default_query" # 파일명 저장을 위한 현재 쿼리 저장 변수

        self.bar_fig = go.Figure() # 일반 Figure 사용
        self.bar_fig.update_layout(
            title='검색 결과 관련성 (%)',
            height=260,
            margin=dict(t=40)
        )
        self.bar_fig.add_trace(go.Bar(x=[], y=[], text=[], textposition='auto'))

    def _init_scene(self):
        """3D 장면 초기화: 질문 중심 배치"""
        # 1. 쿼리 노드 (중앙, 빨간색 다이아몬드)
        self.query_trace = go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers+text',
            marker=dict(
                size=5,
                color='red',
                symbol='diamond',
                line=dict(width=2, color='darkred')
            ),
            text=['❓ Query'],
            textposition='bottom center',
            name='사용자 질문',
            showlegend=True
        )
        self.fig3d.add_trace(self.query_trace)
        
        # 2. 검색 결과 노드
        self.search_trace = go.Scatter3d(
            x=[], y=[], z=[],
            mode='markers+text',
            marker=dict(
                size=[], 
                color=[], 
                colorscale='Viridis',
                cmin=0, cmax=100, 
                opacity=0.9, 
                showscale=True,
                colorbar=dict(title='관련성 (%)')
            ),
            text=[], 
            textposition='top center',
            name='검색 결과'
        )
        self.fig3d.add_trace(self.search_trace)
        
        # 3. 쿼리-결과 연결선 (빨간 점선)
        self.query_edge_trace = go.Scatter3d(
            x=[], y=[], z=[],
            mode='lines',
            line=dict(color='rgba(255,0,0,0.7)', width=2, dash='dot'),
            name='쿼리-문서 연결',
            showlegend=False
        )
        self.fig3d.add_trace(self.query_edge_trace)
        
        # 4. 레이아웃 설정 (축 숨김)
        self.fig3d.update_layout(
            title='🔍 RAG: 질문 기반 문서 검색 (가까울수록 유사도 ↑)',
            scene=dict(
                xaxis=dict(visible=False, showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(visible=False, showticklabels=False, showgrid=False, zeroline=False),
                zaxis=dict(title='거리 (관련성 ↓)', showgrid=True),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=700,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

    def show_visualization(self):
        """스크립트 실행 후 브라우저에 그래프를 띄웁니다."""
        
        # 파일명 생성: 쿼리 내용을 기반으로 파일명 생성
        # 특수문자 제거 및 공백을 언더스코어로 변경하여 안전한 파일명 생성
        safe_query = re.sub(r'[\\/*?:"<>|]', "", self.current_query_for_filename) # 윈도우 파일명 금지 문자 제거
        safe_query = safe_query.strip().replace(" ", "_")
        if len(safe_query) > 50: # 파일명이 너무 길어지지 않도록 제한
            safe_query = safe_query[:50]
            
        # 저장 폴더 설정
        output_dir = "search"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        filename = os.path.join(output_dir, f"{safe_query}_3d_visualization.html")
        
        # 3D 그래프 표시
        print(f"\n[INFO] 3D 시각화 결과를 '{filename}' 파일로 저장하고 웹 브라우저에서 표시합니다.")
        pyo.plot(self.fig3d, filename=filename, auto_open=True)
        
        # 막대 그래프 표시 (선택 사항: 3D만 필요하다면 이 부분을 제거하거나 주석 처리하세요.)
        bar_filename = os.path.join(output_dir, f"{safe_query}_bar_chart.html")
        pyo.plot(self.bar_fig, filename=bar_filename, auto_open=True)
        print(f"[INFO] '{bar_filename}' 파일이 생성되어 브라우저에서 열렸습니다.")


    def update_search_results(self, results: List[Dict[str, Any]], current_query: str):
        """검색 결과를 3D 그래프와 막대 차트에 업데이트"""
        if not results: return
            
        self._update_query_text(current_query)
        self.current_query_for_filename = current_query # 파일명 생성을 위해 쿼리 저장

        res_sorted = sorted(results, key=lambda x: x.get("relevance", 0), reverse=True)
        n = len(res_sorted)
        
        bx, by, bz = self.query_position
        min_dist = 0.5
        max_dist = 4.0
        
        xs, ys, zs = [], [], []
        edge_xs, edge_ys, edge_zs = [], [], []
        sizes, colors, texts = [], [], []
        golden_angle = np.pi * (3 - np.sqrt(5))

        for i, item in enumerate(res_sorted):
            rel = float(item.get("relevance", 0.0))
            dist = min_dist + (1 - rel / 100) * (max_dist - min_dist)
            
            theta = i * golden_angle
            phi = np.arccos(1 - 2 * (i + 0.5) / max(n, 1))
            
            x = bx + dist * np.sin(phi) * np.cos(theta)
            y = by + dist * np.sin(phi) * np.sin(theta)
            z = bz + dist * np.cos(phi)
            
            xs.append(x); ys.append(y); zs.append(z)
            edge_xs.extend([bx, x, None]); edge_ys.extend([by, y, None]); edge_zs.extend([bz, z, None])
            sizes.append(max(8, rel / 4.0))
            colors.append(rel)
            texts.append(f"{item.get('file_name','문서')}<br>{rel}%")

        # 3D 그래프 업데이트 (일반 Figure는 data 속성을 직접 할당)
        self.fig3d.data[1].x = xs
        self.fig3d.data[1].y = ys
        self.fig3d.data[1].z = zs
        self.fig3d.data[1].marker.size = sizes
        self.fig3d.data[1].marker.color = colors
        self.fig3d.data[1].text = texts
        
        self.fig3d.data[2].x = edge_xs
        self.fig3d.data[2].y = edge_ys
        self.fig3d.data[2].z = edge_zs
        
        # 막대 차트 업데이트
        res_sorted_bar = sorted(results, key=lambda x: x.get("relevance", 0), reverse=True)
        file_names_bar = [res['file_name'] for res in res_sorted_bar]
        relevances_bar = [res['relevance'] for res in res_sorted_bar]
        
        # 일반 Figure는 data[0] 속성을 직접 수정
        self.bar_fig.data[0].x = file_names_bar
        self.bar_fig.data[0].y = relevances_bar
        self.bar_fig.data[0].text = [f"{r}%" for r in relevances_bar]
        self.bar_fig.data[0].marker.color = relevances_bar
        self.bar_fig.data[0].marker.colorscale = 'Viridis'

    def _update_query_text(self, query: str):
        q_text = f"❓ {query[:30]}" + ("..." if len(query) > 30 else "")
        self.fig3d.data[0].text = [q_text]


# ================================================================
# 5. 시각화 객체 생성
# ================================================================
# RAGVisualizerScript로 변경
visualizer = RAGVisualizerScript()

# ================================================================
# 6. RAG 에이전트 함수 (프로세스 로직 유지)
# ================================================================
def extractor_agent(state: AgentState):
    visualizer._update_query_text(state["query"])
    # 쿼리가 처음 들어올 때 파일명 생성을 위해 저장해둡니다.
    visualizer.current_query_for_filename = state["query"] 
    
    prompt = PromptTemplate.from_template(
        "사용자의 질문에서 띄어쓰기 확인하고 찾고자 하는 키워드를 쉼표로 구분하여 출력하세요. "
        "벡터스토어 검색을 위한 최적의 키워드를 추출해주세요.\n질문: {query}"
    )
    keywords = LLM.invoke(prompt.format(query=state["query"]))
    
    return {**state, "keywords": keywords.strip()}

# rag_search_agent 함수 내부의 관련성 계산 부분을 수정
def rag_search_agent(state: AgentState):
    try:
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDINGS)
        results = vectorstore.similarity_search_with_score(state["keywords"], k=10)

        if not results: return {**state, "search_results": [], "context": ""}
        
        best_doc_info = {}
        for doc, score in results:
            doc_id = doc.metadata.get("id")
            if doc_id:
                content = doc.page_content.strip()
                if content:
                    if doc_id not in best_doc_info or score < best_doc_info[doc_id][0]:
                        best_doc_info[doc_id] = (score, content)
        
        if not best_doc_info: return {**state, "search_results": [], "context": ""}
        
        scores = [score for score, _ in best_doc_info.values()]
        min_score, max_score = min(scores), max(scores)
        
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        search_results_with_metadata = []
        
        for doc_id, (score, content) in best_doc_info.items():
            cursor.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
            row = cursor.fetchone()
            if row:
                # 관련성 계산: 0-100% 범위로 정규화한 후 98%로 제한
                if max_score != min_score:
                    relevance = (1 - (score - min_score) / (max_score - min_score)) * 100
                else:
                    relevance = 100.0
                
                # 관련성이 98%를 넘지 않도록 제한
                relevance = min(relevance, 98.0)
                
                search_results_with_metadata.append({
                    "relevance": round(relevance, 1),  # 소수점 한 자리로 반올림
                    "file_name": row["file_name"],
                    "file_location": row["file_location"],
                    "summary": row["summary"][:100] + "..." if row["summary"] else "요약 없음",
                    "doc_type": row["doc_type"],
                    "keywords": row["keywords"],
                    "content": content
                })
        conn.close()
        
        search_results_with_metadata.sort(key=lambda x: x["relevance"], reverse=True)
        context = "\n\n".join([res['content'] for res in search_results_with_metadata[:10]])
        
        # 시각화 업데이트
        visualizer.update_search_results(search_results_with_metadata, state["query"])
        
        return {**state, "search_results": search_results_with_metadata, "context": context}
    
    except Exception as e:
        print(f"❌ RAG 검색 오류: {e}", file=sys.stderr)
        visualizer.update_search_results([], state["query"]) 
        return {**state, "search_results": [], "context": ""}

def answer_generator_agent(state: AgentState):
    if not state["search_results"]:
        return {**state, "result": "관련 정보 없음"}
    
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
    if not state["search_results"]:
        return state
    
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
# 7. LangGraph 구성
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
# 8. 실행 예시 (Python Script Output)
# ================================================================
# 1) RAG 파이프라인 실행
state = {"query": "카카오 뱅크 관련 내용", "keywords": "", "search_results": [], "context": "", "result": ""}
result = app.invoke(state)

# 2) 최종 결과 출력
print("\n" + "="*50)
print("📊 RAG 처리 완료! 최종 결과:")
print("="*50)
print(result["result"])

# 3) 시각화 표시 (스크립트 실행 시 브라우저 자동 실행)
visualizer.show_visualization()