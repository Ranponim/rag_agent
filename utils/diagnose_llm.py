import os
import sys
from pathlib import Path
import httpx
import logging

# 프로젝트 루트를 path에 추가하여 config 로드
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from langchain_openai import ChatOpenAI

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diagnosis")

def diagnose():
    settings = get_settings()
    
    print("\n" + "="*60)
    print("🔍 LLM 연결 진단 스크립트")
    print("="*60)
    
    api_key = settings.openai_api_key or "dummy-key"
    base_url = settings.openai_api_base
    
    print(f"📍 Target Base URL: {base_url}")
    print(f"🔑 API Key: {api_key[:4]}***")
    
    # 1. 단순 HTTP 연결 (Models 엔드포인트)
    print("\n[1] 기본 HTTP 연결 테스트 (/v1/models)")
    try:
        models_url = f"{base_url.rstrip('/')}/models"
        print(f"   URL: {models_url}")
        
        headers = {"Authorization": f"Bearer {api_key}"}
        response = httpx.get(models_url, headers=headers, timeout=5.0)
        
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
            model=settings.openai_model,
            temperature=0,
            max_retries=1, # 빠른 실패를 위해
        )
        
        print(f"   LLM 생성: {llm}")
        print("   메시지 전송 중...") 
        
        # 클라이언트의 실제 Base URL 확인
        if hasattr(llm, "client"):
             print(f"   OpenAI Client Base URL: {llm.client.base_url}")
        
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

if __name__ == "__main__":
    diagnose()
