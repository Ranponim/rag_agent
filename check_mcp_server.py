import asyncio
import httpx
import json
import sys
import requests
import time

# PowerShell 성공 사례 헤더 재현
COMMON_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
    "Connection": "keep-alive"
}

# MCP Initialize 페이로드 (PowerShell $body 추정값)
INITIALIZE_PAYLOAD = {
    "jsonrpc": "2.0",
    "method": "initialize",
    "id": 1,
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "mcp-checker", "version": "1.0.0"}
    }
}

def test_with_requests(url):
    print(f"\n--- [Requests Sync Test] {url} ---")
    try:
        with requests.post(url, headers=COMMON_HEADERS, json=INITIALIZE_PAYLOAD, stream=True, timeout=10) as r:
            print(f"✅ Status: {r.status_code} {r.reason}")
            print(f"📝 Headers: {dict(r.headers)}")
            # 첫 번째 데이터 라인만 읽어봄
            count = 0
            for line in r.iter_lines():
                if line:
                    print(f"📊 Received: {line.decode('utf-8')}")
                    count += 1
                    if count >= 3: break # 너무 많이 읽지 않음
            if count > 0:
                print("🎉 SUCCESS: Successfully received SSE data using requests!")
    except Exception as e:
        print(f"❌ Requests Failed: {type(e).__name__}: {e}")

async def test_with_httpx(url):
    print(f"\n--- [Httpx Async Test] {url} ---")
    try:
        async with httpx.AsyncClient(http1=True, http2=False, timeout=10.0) as client:
            print("📡 Sending POST with stream=True...")
            async with client.stream("POST", url, headers=COMMON_HEADERS, json=INITIALIZE_PAYLOAD) as response:
                print(f"✅ Status: {response.status_code}")
                print(f"📝 Headers: {dict(response.headers)}")
                count = 0
                async for line in response.aiter_lines():
                    if line.strip():
                        print(f"📊 Received: {line}")
                        count += 1
                        if count >= 3: break
                if count > 0:
                    print("🎉 SUCCESS: Successfully received SSE data using httpx!")
    except httpx.RemoteProtocolError as e:
        print(f"❌ Httpx RemoteProtocolError: {e}")
        print("💡 분석: 서버가 응답을 보내기 전에 연결을 끊었습니다. (Keep-alive 또는 포맷 이슈)")
    except Exception as e:
        print(f"❌ Httpx Failed: {type(e).__name__}: {e}")

async def main():
    target_url = "http://165.213.69.30:8001/mcp"
    local_url = "http://localhost:8001/mcp"
    
    # 원격 테스트
    print("🚀 [REMOTE] Testing 165.213.69.30:8001...")
    test_with_requests(target_url)
    await test_with_httpx(target_url)
    
    # 로컬 테스트
    print("\n🚀 [LOCAL] Testing localhost:8001...")
    test_with_requests(local_url)
    await test_with_httpx(local_url)

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
