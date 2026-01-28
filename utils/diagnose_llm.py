import os
import sys
from pathlib import Path
import httpx
import logging

# .env 파일에서 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# 프로젝트 루트를 path에 추가하여 유틸리티 로드
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diagnosis")

def diagnose():
    # 환경변수에서 설정 로드
    api_key = os.getenv("OPENAI_API_KEY", "lm-studio")
    base_url = os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
    model = os.getenv("OPENAI_MODEL", "local-model")
    
    print("\n" + "="*60)
    print("🔍 LLM 연결 진단 스크립트")
    print("="*60)
    
    print(f"📍 Target Base URL: {base_url}")
    print(f"🔑 API Key: {api_key[:4]}***")
    
    # 1. 단순 HTTP 연결 (Models 엔드포인트)
    print("\n[1] 기본 HTTP 연결 테스트 (/v1/models)")
    try:
        models_url = f"{base_url.rstrip('/')}/models"
        print(f"   URL: {models_url}")
        
        # Curl과 유사한 헤더 설정
        headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "curl/7.83.1", # Mimic curl
            "Accept": "*/*"
        }
        
        # trust_env=False로 시스템 프록시 무시, verifying=False로 인증서 무시 시도
        transport = httpx.HTTPTransport(retries=1)
        with httpx.Client(transport=transport, trust_env=False, verify=False) as client:
            response = client.get(models_url, headers=headers, timeout=10.0)
        
        print(f"   상태 코드: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ 연결 성공!")
            try:
                data = response.json()
                print(f"   모델 목록: {[m['id'] for m in data.get('data', [])[:3]]} ...")
            except:
                print(f"   응답 본문: {response.text[:100]}...")
        else:
            print(f"   ❌ 연결 실패 (HTTP {response.status_code})")
            print(f"   응답 본문: {response.text}")
            
    except Exception as e:
        print(f"   ❌ 예외 발생: {e}")

    # 2. LangChain ChatOpenAI 테스트
    print("\n[2] LangChain ChatOpenAI 테스트")
    try:
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0,
            max_retries=1, # 빠른 실패를 위해
        )
        
        print(f"   LLM 생성: {llm}")
        print("   메시지 전송 중...") 
        
        # 클라이언트의 실제 Base URL 확인
        try:
            if hasattr(llm, "client") and hasattr(llm.client, "base_url"):
                 print(f"   OpenAI Client Base URL: {llm.client.base_url}")
            elif hasattr(llm, "base_url"):
                 print(f"   ChatOpenAI Base URL: {llm.base_url}")
        except Exception as e:
            print(f"   (Base URL 확인 불가: {e})")
        
        response = llm.invoke("Hello, simple test.")
        print(f"   ✅ 응답 수신: {response.content}")
        
    except Exception as e:
        print(f"   ❌ LangChain 오류: {e}")
        import traceback
        traceback.print_exc()

    # 3. Typo Check (Completitions)
    print("\n[3] Typo URL Check (/chat/completitions)")
    try:
        typo_url = f"{base_url.rstrip('/')}/chat/completitions"
        print(f"   Testing Typo URL: {typo_url}")
        response = httpx.post(typo_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=2.0)
        print(f"   상태 코드: {response.status_code}")
        if response.status_code != 404:
             print("   ⚠️ WARNING: 서버가 오타난 URL(/chat/completitions)에 응답했습니다!")
    except Exception as e:
        print(f"   (오타 URL 연결 실패 - 정상: {e})")

    # 4. LangChain Streaming 테스트
    print("\n[4] LangChain 스트리밍 테스트")
    try:
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0,
            max_retries=1,
            streaming=True
        )
        
        print("   스트리밍 시작 (응답이 한 글자씩 표시되어야 함):")
        print("   > ", end="", flush=True)
        
        for chunk in llm.stream("Tell me a short sentence about why coding is fun."):
            content = chunk.content
            if content:
                print(content, end="", flush=True)
        
        print("\n   ✅ 스트리밍 완료!")
            
    except Exception as e:
        print(f"\n   ❌ 스트리밍 오류: {e}")

if __name__ == "__main__":
    diagnose()
