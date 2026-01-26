import requests
import httpx
import json
import os
import sys

# PowerShell 성공 사례를 바탕으로 한 헤더 및 페이로드 설정
COMMON_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
    "Connection": "keep-alive"
}

INITIALIZE_PAYLOAD = {
    "jsonrpc": "2.0",
    "method": "initialize",
    "id": 1,
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "1.0.0"}
    }
}

LIST_TOOLS_PAYLOAD = {
    "jsonrpc": "2.0",
    "method": "list_tools",
    "id": 2
}

def test_requests(url):
    print(f"\n🚀 [Requests Test] Target: {url}")
    # 프록시를 강제로 비활성화하여 환경 격리
    proxies = {"http": None, "https": None}
    
    try:
        session = requests.Session()
        session.proxies = proxies
        
        print("1. Sending 'initialize'...")
        resp = session.post(url, headers=COMMON_HEADERS, json=INITIALIZE_PAYLOAD, timeout=10)
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"   Response: {resp.text[:200]}...")
            
            print("2. Sending 'list_tools'...")
            resp = session.post(url, headers=COMMON_HEADERS, json=LIST_TOOLS_PAYLOAD, timeout=10)
            print(f"   Status: {resp.status_code}")
            print(f"   Response: {resp.text[:200]}...")
        else:
            print(f"   Failed to initialize: {resp.text}")
            
    except Exception as e:
        print(f"❌ Requests Failed: {type(e).__name__}: {e}")

async def test_httpx(url):
    print(f"\n🚀 [Httpx Test] Target: {url}")
    # httpx에서 프록시 무시 및 HTTP/1.1 강제
    try:
        async with httpx.AsyncClient(
            trust_env=False, # 환경변수 프록시 무시
            http1=True,
            http2=False,
            headers=COMMON_HEADERS,
            timeout=10.0
        ) as client:
            print("1. Sending 'initialize'...")
            resp = await client.post(url, json=INITIALIZE_PAYLOAD)
            print(f"   Status: {resp.status_code}")
            if resp.status_code == 200:
                print(f"   Response: {resp.text[:200]}...")
                
                print("2. Sending 'list_tools'...")
                resp = await client.post(url, json=LIST_TOOLS_PAYLOAD)
                print(f"   Status: {resp.status_code}")
                print(f"   Response: {resp.text[:200]}...")
            else:
                print(f"   Failed: {resp.text}")
                
    except httpx.RemoteProtocolError as e:
        print(f"❌ Httpx RemoteProtocolError: {e}")
    except Exception as e:
        print(f"❌ Httpx Failed: {type(e).__name__}: {e}")

async def main():
    # 165 IP와 localhost 두가지 모두 테스트
    targets = [
        "http://165.213.69.30:8001/mcp",
        "http://localhost:8001/mcp"
    ]
    
    for url in targets:
        print(f"\n{'='*50}\n🔍 Testing {url}\n{'='*50}")
        test_requests(url)
        import asyncio
        await test_httpx(url)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
