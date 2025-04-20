import streamlit as st
from utils.time_utils import get_brasilia_time
from datetime import datetime, timedelta
import google.cloud.firestore as firestore
from rag.gemini_integration import generate_response_with_gemini

def show_chat_interface(query_engine, firestore_db, chat_id, messages):
    """Exibe a interface principal do chatbot"""
    
    st.title(f"🤖 IAHC ChatBot")
    st.info("Este chatbot utiliza RAG (Retrieval Augmented Generation) para fornecer respostas precisas sobre IHC.", icon="ℹ️")
    
    # Exibir mensagens anteriores
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Campo de entrada para nova mensagem
    if prompt := st.chat_input("Faça uma pergunta sobre IHC"):
        # O chat agora tem mensagens, não é mais temporário
        if "is_temp_chat" in st.session_state:
            st.session_state.is_temp_chat = False
            
        # Adicionar nova mensagem ao estado
        new_message = {
            "role": "user", 
            "content": prompt,
            "timestamp": get_brasilia_time(),
            "saved": False
        }
        messages.append(new_message)
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Verificar se é a primeira mensagem do usuário e atualizar o título do chat
        is_first_message = len([m for m in messages if m["role"] == "user"]) == 1
        
        # Atualizar título do chat se for a primeira mensagem
        if is_first_message and chat_id:
            # Limitar o tamanho do título para evitar títulos muito longos
            if len(prompt) > 50:
                new_chat_title = prompt[:47] + "..."
            else:
                new_chat_title = prompt
                
            # Atualizar título no Firestore e no estado da sessão
            chat_ref = firestore_db.collection("chats").document(chat_id)
            chat_ref.update({"title": new_chat_title})
            
            # Atualizar no estado da sessão
            if "chats" in st.session_state and chat_id in st.session_state.chats:
                st.session_state.chats[chat_id]["title"] = new_chat_title
        
        with st.status("Processando sua pergunta...", expanded=True) as status:
            st.write("Buscando informações relevantes...")
            
            # Verifica se o usuário especificou um artigo: 'nome.txt'
            file_match = re.search(r"artigo:\s*[\"'](.+?\.txt)[\"']", prompt, re.IGNORECASE)
        
            if file_match:
                file_name_filter = file_match.group(1)
        
                # Obter todos os nós do índice (baseados no docstore interno)
                all_nodes = query_engine._index.docstore.get_all_nodes()
                
                # Filtrar nós que pertencem ao arquivo citado
                nodes_from_file = [
                    node for node in all_nodes
                    if file_name_filter.lower() in node.metadata.get("file_name", "").lower()
                ]
        
                # Criar um retriever personalizado só com os chunks do arquivo citado
                custom_retriever = VectorStoreIndex(nodes_from_file).as_retriever(similarity_top_k=10)
                retrieved_nodes = custom_retriever.retrieve(prompt)
            else:
                # Caso nenhum artigo específico tenha sido citado, busca geral
                retrieved_nodes = query_engine.retrieve(prompt)
        
            # Filtrar e ordenar nós com bom score
            filtered_nodes = sorted(
                [n for n in retrieved_nodes if n.score >= 0.6],
                key=lambda x: x.score,
                reverse=True
            )[:7]  # Pega os 7 chunks mais relevantes
        
            # Construir o contexto para o prompt
            context = "\n\n".join(
                f"📄 Fonte: {n.metadata.get('file_name', 'Desconhecida')}\n"
                f"📊 Score: {n.score:.2f}\n"
                f"🔍 Conteúdo:\n{n.node.text[:1000]}..."
                for n in filtered_nodes
            )
            
            st.write("Gerando resposta...")
            full_prompt = f"""Você é um especialista em IHC (Interação Humano-Computador) com vasta experiência acadêmica e prática.

            [INSTRUÇÕES]
            1. Analise cuidadosamente a pergunta e o contexto fornecido.
            2. Se o contexto contiver informações relevantes para a pergunta, baseie sua resposta principalmente nessas informações.
            3. Se o contexto for insuficiente ou não abordar diretamente a pergunta, forneça uma resposta baseada em seu conhecimento geral de IHC, sem mencionar a ausência de informações no contexto.
            4. Não faça referências diretas aos \"textos fornecidos\" ou \"artigos\" na sua resposta.
            
            [FORMATO]
            - Use português brasileiro formal
            - Mantenha termos técnicos consolidados em inglês quando apropriado
            - Estruture sua resposta em parágrafos claros e concisos
            - Inclua exemplos práticos quando relevante
            - Apresente diferentes perspectivas quando apropriado
            
            [CONTEXTO]
            {context}
            
            Pergunta: {prompt}
            """
            response = generate_response_with_gemini(full_prompt)
            
            if response:
                response = response.replace('[PERGUNTA]', '').replace('[RESPOSTA]', '').strip()
                
                # Adicionar resposta ao estado
                new_response = {
                    "role": "assistant", 
                    "content": response,
                    "timestamp": get_brasilia_time(),
                    "saved": False
                }
                messages.append(new_response)
                
                with st.chat_message("assistant"):
                    st.write(response)
                
                # Salvar chat e mensagens no Firestore
                save_chat_to_firestore(firestore_db, chat_id, messages)
            
            status.update(label="Resposta gerada!", state="complete", expanded=True)

def save_chat_to_firestore(firestore_db, chat_id, messages):
    """
    Salva o chat e suas mensagens no Firestore
    
    Args:
        firestore_db: Cliente Firestore
        chat_id: ID do chat
        messages: Lista de mensagens
    """
    if not chat_id or "chats" not in st.session_state or chat_id not in st.session_state.chats:
        return
    
    # Atualizar timestamp do chat
    now = get_brasilia_time()
    chat_data = {
        "updated_at": now,
    }
    
    # Atualizar documento principal do chat
    chat_ref = firestore_db.collection("chats").document(chat_id)
    chat_ref.update(chat_data)
    
    # Adicionar mensagens não salvas à subcoleção
    for idx, msg in enumerate(messages):
        if not msg.get("saved", False):  # Apenas salva mensagens não salvas anteriormente
            msg_data = {
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg.get("timestamp", now),
                "order": idx  # Para manter a ordem das mensagens
            }
            chat_ref.collection("messages").add(msg_data)
            msg["saved"] = True  # Marca como salva

def load_chat_messages_from_firestore(firestore_db, chat_id):
    """
    Carrega as mensagens de um chat do Firestore
    
    Args:
        firestore_db: Cliente Firestore
        chat_id: ID do chat
        
    Returns:
        Lista de mensagens ordenadas
    """
    messages = []
    
    if not chat_id:
        return messages
    
    # Referência para a coleção de mensagens do chat
    message_refs = firestore_db.collection("chats").document(chat_id) \
                     .collection("messages") \
                     .order_by("order") \
                     .stream()
    
    # Converter documentos para o formato esperado
    for msg in message_refs:
        msg_data = msg.to_dict()
        messages.append({
            "role": msg_data["role"],
            "content": msg_data["content"],
            "timestamp": msg_data.get("timestamp", get_brasilia_time()),
            "saved": True  # Marcar como já salva no Firestore
        })
    
    return messages

def create_new_chat_in_firestore(firestore_db, user_id, title):
    """
    Cria um novo chat no Firestore
    
    Args:
        firestore_db: Cliente Firestore
        user_id: ID do usuário
        title: Título do chat
        
    Returns:
        ID do chat criado
    """
    now = get_brasilia_time()
    
    # Dados do novo chat
    chat_data = {
        "user_id": user_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "expiry_date": now + timedelta(days=30)  # Define data de expiração para 30 dias
    }
    
    # Adicionar documento à coleção de chats
    chat_ref = firestore_db.collection("chats").document()
    chat_ref.set(chat_data)
    
    return chat_ref.id

def list_user_chats_from_firestore(firestore_db, user_id):
    """
    Lista todos os chats de um usuário
    
    Args:
        firestore_db: Cliente Firestore
        user_id: ID do usuário
        
    Returns:
        Dicionário de chats com ID como chave
    """
    chats = {}
    
    # Consultar chats do usuário
    chat_refs = firestore_db.collection("chats").where("user_id", "==", user_id)
    chat_refs = chat_refs.order_by("updated_at", direction="DESCENDING")
    chat_refs = chat_refs.stream()
    
    # Converter para dicionário
    for chat in chat_refs:
        chat_data = chat.to_dict()
        chats[chat.id] = chat_data
    
    return chats

def cleanup_old_chats(firestore_db, days=30):
    """
    Remove chats e mensagens mais antigos que o número especificado de dias
    
    Args:
        firestore_db: Cliente Firestore
        days: Número de dias para manter os chats (padrão: 30)
    """
    now = get_brasilia_time()
    cutoff_date = now - timedelta(days=days)
    
    # Encontrar chats expirados
    expired_chats = firestore_db.collection("chats") \
                      .where("updated_at", "<", cutoff_date) \
                      .stream()
    
    batch_size = 0
    batch = firestore_db.batch()
    
    for chat in expired_chats:
        chat_ref = firestore_db.collection("chats").document(chat.id)
        
        # Excluir todas as mensagens primeiro
        messages = chat_ref.collection("messages").stream()
        for msg in messages:
            msg_ref = chat_ref.collection("messages").document(msg.id)
            batch.delete(msg_ref)
            batch_size += 1
            
            # Enviar batch quando atingir limite (500 é o máximo para firestore)
            if batch_size >= 450:
                batch.commit()
                batch = firestore_db.batch()
                batch_size = 0
        
        # Excluir o documento do chat
        batch.delete(chat_ref)
        batch_size += 1
        
        # Enviar batch quando atingir limite
        if batch_size >= 450:
            batch.commit()
            batch = firestore_db.batch()
            batch_size = 0
    
    # Enviar batch final se houver operações pendentes
    if batch_size > 0:
        batch.commit()
    
    print(f"Limpeza de chats antigos concluída: {now}")
