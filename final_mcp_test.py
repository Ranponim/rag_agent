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
    "params": {}, # 명시적으로 빈 params 추가
    "id": 2
}

def test_requests(url):
    print(f"\n🚀 [Requests Test] Target: {url}")
    proxies = {"http": None, "https": None}
    
    try:
        session = requests.Session()
        session.proxies = proxies
        
        print(f"1. Sending 'initialize' to {url}...")
        resp = session.post(url, headers=COMMON_HEADERS, json=INITIALIZE_PAYLOAD, timeout=15)
        print(f"   Status: {resp.status_code}")
        
        # mcp-session-id 추출
        session_id = resp.headers.get("mcp-session-id")
        print(f"   mcp-session-id: {session_id}")
        
        if resp.status_code == 200:
            custom_headers = COMMON_HEADERS.copy()
            if session_id:
                custom_headers["mcp-session-id"] = session_id
                print(f"   Applying mcp-session-id to next request...")
            
            print(f"\n2. Sending 'list_tools' with session-id...")
            resp = session.post(url, headers=custom_headers, json=LIST_TOOLS_PAYLOAD, timeout=15)
            print(f"   Status: {resp.status_code}")
            print(f"   Full Response: {resp.text}")
        else:
            print(f"   Failed to initialize.")
            
    except Exception as e:
        print(f"❌ Requests Failed: {type(e).__name__}: {e}")

async def test_httpx(url):
    print(f"\n🚀 [Httpx Test] Target: {url}")
    try:
        async with httpx.AsyncClient(
            trust_env=False, 
            http1=True,
            http2=False,
            headers=COMMON_HEADERS,
            timeout=15.0
        ) as client:
            print(f"1. Sending 'initialize' to {url}...")
            resp = await client.post(url, json=INITIALIZE_PAYLOAD)
            print(f"   Status: {resp.status_code}")
            
            session_id = resp.headers.get("mcp-session-id")
            print(f"   mcp-session-id: {session_id}")
            
            if resp.status_code == 200:
                headers = COMMON_HEADERS.copy()
                if session_id:
                    headers["mcp-session-id"] = session_id
                
                print(f"\n2. Sending 'list_tools' with session-id...")
                resp = await client.post(url, json=LIST_TOOLS_PAYLOAD, headers=headers)
                print(f"   Status: {resp.status_code}")
                print(f"   Full Response: {resp.text}")
            else:
                print(f"   Failed to initialize.")
                
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
