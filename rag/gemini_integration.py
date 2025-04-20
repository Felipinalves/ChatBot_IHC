import google.generativeai as genai
import time
import streamlit as st

# Configuração da API do Gemini
GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]

def generate_response_with_gemini(prompt, max_retries=3):
    """
    Gera uma resposta usando o modelo Gemini da Google
    
    Args:
        prompt: Texto de entrada para o modelo
        max_retries: Número máximo de tentativas em caso de erro
        
    Returns:
        Texto da resposta ou None em caso de erro
    """
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }
        
        model = genai.GenerativeModel(
            model_name="gemini-2.0-pro-exp-02-05",
            generation_config=generation_config
        )
        
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                if response and hasattr(response, 'text'):
                    return response.text
                return None
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                st.error(f"Erro ao gerar resposta: {str(e)}")
                return None
    except Exception as e:
        st.error(f"Erro na configuração do Gemini: {str(e)}")
        return "Não foi possível conectar ao modelo de linguagem. Por favor, verifique a configuração da API."
