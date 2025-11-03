import os
import pymysql
import re
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# DB 접속 정보
DB_CONFIG = {
    'host': 'localhost',
    'user': 'admin',
    'password': '1qazZAQ!',
    'db': 'final',
    'charset': 'utf8mb4'
}

# 프롬프트 템플릿
PROMPT_TEMPLATE = """
다음 문서 내용을 분석하여 아래 항목을 한글로 한 줄씩 추출해줘.

title: 문서의 제목을 한 줄로,
summary: 전체 내용을 1000자 이내로 줄거리처럼 요약해줘. (띄어쓰기 포함 1000자 이하, 너무 짧게 쓰지 말고 최대한 자세히)
keywords: 문서의 핵심 단어를 50개 정도, 쉼표(,)로 구분해서 한 줄로 나열해줘.

아래 형식으로만 출력해줘.
title: ...
summary: ...
keywords: ...

문서 내용:
"{text}"
"""

CUSTOM_PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["text"])

# 파일 확장자 추출 함수
def get_doc_type(file_name):
    """파일명에서 확장자 추출 (.포함)"""
    ext = os.path.splitext(file_name)[1].lower()
    return ext if ext else ".unknown"

# 파일 불러오기 함수
def load_document(file_path: str):
    """안정적인 문서 로더"""
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(file_path)
            
        elif ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            
        elif ext == ".csv":
            from langchain_community.document_loaders import CSVLoader
            loader = CSVLoader(file_path, encoding="utf-8")
            
        elif ext == ".txt":
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file_path, encoding="utf-8")
            
        elif ext in (".html", ".htm"):
            from langchain_community.document_loaders import UnstructuredHTMLLoader
            loader = UnstructuredHTMLLoader(file_path)
            
        else:
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file_path, encoding="utf-8")
            
        docs = loader.load()
        
        # 빈 문서 필터링
        docs = [doc for doc in docs if doc.page_content.strip()]
        
        return docs
        
    except Exception as e:
        print(f"⚠️ {file_path} 로더 에러: {e}")
        return []

# LLM 요약 함수 (LangChain v0.2+ 호환)
def summarize_with_llm(docs):
    """문서 요약 처리"""
    try:
        llm = Ollama(model="exaone3.5:2.4b")
        
        # 문서가 많으면 맨 앞 5개만 사용
        if len(docs) > 5:
            docs = docs[:5]
            print(f"📄 문서 청크가 많아 상위 5개만 처리합니다.")
        
        # 문서 내용 결합
        combined_content = "\n\n".join([doc.page_content for doc in docs])
        
        # Prompt + LLM 조합 (Runnable Sequence)
        chain = CUSTOM_PROMPT | llm
        
        # 실행
        result = chain.invoke({"text": combined_content})
        
        return result  # 문자열 그대로 반환
        
    except Exception as e:
        print(f"⚠️ LLM 요약 에러: {e}")
        return ""

# LLM 결과 파싱 함수
def parse_llm_output(output_text):
    """LLM 출력 파싱"""
    try:
        title_match = re.search(r"title\s*:\s*(.+?)(?=\n|summary:|$)", output_text, re.DOTALL)
        summary_match = re.search(r"summary\s*:\s*(.+?)(?=\n|keywords:|$)", output_text, re.DOTALL)
        keywords_match = re.search(r"keywords\s*:\s*(.+?)(?=\n|$)", output_text, re.DOTALL)
        
        result = {
            "title": title_match.group(1).strip() if title_match else "제목 없음",
            "summary": summary_match.group(1).strip() if summary_match else "요약 없음",
            "keywords": keywords_match.group(1).strip() if keywords_match else "키워드 없음",
        }
        
        if len(result["summary"]) > 1000:
            result["summary"] = result["summary"][:997] + "..."
            
        return result
        
    except Exception as e:
        print(f"⚠️ 파싱 에러: {e}")
        return {
            "title": "파싱 실패",
            "summary": "요약 생성 실패",
            "keywords": "키워드 추출 실패"
        }

# DB에 INSERT 함수
def insert_into_db(title, summary, keywords, file_location, file_name, doc_type):
    """DB 저장"""
    conn = None
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            sql = """
            INSERT INTO documents
            (title, summary, keywords, file_location, file_name, doc_type, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """
            cursor.execute(sql, (title, summary, keywords, file_location, file_name, doc_type))
        conn.commit()
        print(f"✅ {file_name} DB 저장 완료!")
        print(f"   파일타입: {doc_type}")
        print(f"   제목: {title[:30]}...")
        print(f"   요약: {summary[:50]}...")
        print("=============================================================")
        
    except Exception as e:
        print(f"❌ {file_name} DB 저장 실패: {e}")
        
    finally:
        if conn:
            conn.close()

# 단일 파일 처리 함수
def process_single_file(file_path, file_name):
    """개별 파일 처리"""
    print(f"📄 처리 중: {file_path}")
    
    doc_type = get_doc_type(file_name)
    print(f"📋 파일 타입: {doc_type}")
    
    docs = load_document(file_path)
    if not docs:
        print(f"❌ {file_name} 로드 실패 또는 빈 문서")
        return False
        
    print(f"✅ {file_name} 로드 완료. 청크 수: {len(docs)}")
    
    llm_output = summarize_with_llm(docs)
    if not llm_output:
        print(f"❌ {file_name} LLM 요약 실패")
        return False
        
    parsed = parse_llm_output(llm_output)
    
    insert_into_db(
        title=parsed["title"],
        summary=parsed["summary"],
        keywords=parsed["keywords"],
        file_location=file_path,
        file_name=file_name,
        doc_type=doc_type
    )
    
    return True

# 메인 실행 (단일 파일 지정)
if __name__ == "__main__":
    # ------------------- 여기만 수정 -------------------
    # 처리할 단일 파일 경로 (절대/상대 경로 모두 가능)
    SINGLE_FILE_PATH = "fileList/국민의군대.docx"
    
    if not os.path.exists(SINGLE_FILE_PATH):
        print(f"❌ 파일이 존재하지 않습니다: {SINGLE_FILE_PATH}")
        exit(1)
    
    file_name = os.path.basename(SINGLE_FILE_PATH)
    # ------------------------------------------------
    
    # 필요한 패키지 확인
    try:
        import docx2txt
        print("✅ docx2txt 패키지 확인됨")
    except ImportError:
        print("❌ docx2txt 패키지가 필요합니다: pip install docx2txt")
        exit(1)
        
    try:
        import pypdf
        print("✅ pypdf 패키지 확인됨")
    except ImportError:
        print("❌ pypdf 패키지가 필요합니다: pip install pypdf")
        exit(1)
    
    # 단일 파일 처리
    print(f"🔍 단일 파일 처리 시작: {SINGLE_FILE_PATH}")
    success = process_single_file(SINGLE_FILE_PATH, file_name)
    print(f"\n🎉 처리 완료! 성공: {'예' if success else '아니오'}")