import asyncio
import httpx
import json
import sys

async def check_mcp_post_sse(url):
    print(f"\n🚀 Testing POST-SSE on {url}...")
    
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json"
    }
    
    payload = {
        "jsonrpc": "2.0",
        "method": "list_tools",
        "id": 1
    }
    
    try:
        # 165.213.69.30 주소로 테스트
        async with httpx.AsyncClient(timeout=10.0) as client:
            print(f"📡 Sending POST request with body: {json.dumps(payload)}")
            
            # 스트리밍 응답 처리를 위해 client.stream 사용
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                print(f"✅ Response Status: {response.status_code}")
                print(f"📝 Response Headers: {dict(response.headers)}")
                
                # 첫 번째 chunk 확인
                async for line in response.aiter_lines():
                    if line.strip():
                        print(f"📊 Received: {line}")
                        # 첫 메시지만 확인하고 종료
                        if "data:" in line:
                            break
                            
    except httpx.ConnectTimeout:
        print("❌ Connection Timeout: 서버에 연결할 수 없습니다. (방화벽 또는 서버 다운)")
    except httpx.RemoteProtocolError as e:
        print(f"❌ Remote Protocol Error: {e}")
        print("💡 분석: 서버가 응답 중 연결을 예기치 않게 끊었습니다. 헤더 불일치 또는 서버 측 오류일 수 있습니다.")
    except Exception as e:
        print(f"❌ Unexpected Error: {type(e).__name__}: {e}")

async def main():
    # 사용자의 성공 사례 IP: 165.213.69.30
    remote_url = "http://165.213.69.30:8001/mcp"
    await check_mcp_post_sse(remote_url)
    
    # 로컬 테스트 (사용자가 요청한 환경)
    local_url = "http://localhost:8001/mcp"
    await check_mcp_post_sse(local_url)

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
