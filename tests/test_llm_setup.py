# -*- coding: utf-8 -*-
import unittest
from unittest.mock import patch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings
from utils.llm_factory import get_llm

class TestLLMSetup(unittest.TestCase):
    def test_openai_compatible_setup(self):
        """OpenAI 호환 Base URL 설정 시 LLM 객체가 올바르게 생성되는지 테스트"""
        
        # 가상의 설정값
        mock_settings = Settings(
            openai_api_key="dummy-key",
            openai_api_base="http://localhost:8000/v1",
            openai_model="local-model"
        )
        
        with patch('config.settings.get_settings', return_value=mock_settings):
            # 1. LLM 생성 테스트
            llm = get_llm()
            
            # ChatOpenAI 객체의 속성 확인
            self.assertEqual(llm.model_name, "local-model")
            self.assertEqual(llm.openai_api_base, "http://localhost:8000/v1")
            
            print(f"✅ LLM Model: {llm.model_name}")
            print(f"✅ LLM Base URL: {llm.openai_api_base}")
    
    def test_separate_embedding_url(self):
        """임베딩 전용 URL 설정 테스트"""
        from utils.llm_factory import get_embeddings
        
        mock_settings = Settings(
            openai_api_key="dummy",
            openai_api_base="http://llm:8000/v1",
            openai_embedding_api_base="http://embed:8000/v1"
        )
        
        with patch('config.settings.get_settings', return_value=mock_settings):
            embeddings = get_embeddings()
            # OpenAIEmbeddings는 base_url을 openai_api_base 속성으로 가질 수 있음 (버전에 따라 다름)
            # 또는 client.base_url 확인 필요. 여기서는 객체 생성 시 전달된 인자를 간접 확인하거나,
            # 속성이 있다면 확인. LangChain OpenAIEmbeddings는 openai_api_base 속성을 가짐.
            self.assertEqual(embeddings.openai_api_base, "http://embed:8000/v1")
            print(f"✅ Embedding Base URL: {embeddings.openai_api_base}")

if __name__ == "__main__":
    unittest.main()
