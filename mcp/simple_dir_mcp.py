# -*- coding: utf-8 -*-
import subprocess
from fastmcp import FastMCP

# MCP 서버 생성
mcp = FastMCP("Directory Explorer")

@mcp.tool()
def list_directory_c() -> str:
    """
    C: 드라이브의 디렉토리 및 파일 목록을 보여줍니다.
    'dir c:\\' 명령어를 실행한 결과를 반환합니다.
    """
    try:
        # 윈도우 dir 명령어 실행
        # shell=True가 있어야 dir 같은 쉘 내부 명령어가 실행됨
        result = subprocess.run(
            ["cmd", "/c", "dir", "c:\\"], 
            capture_output=True, 
            text=True, 
            encoding='cp949', # 한글 윈도우 기본 인코딩
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"명령어 실행 중 오류 발생: {e}\n{e.stderr}"
    except Exception as e:
        return f"예기치 않은 오류 발생: {e}"

if __name__ == "__main__":
    # MCP 서버 실행
    mcp.run()
