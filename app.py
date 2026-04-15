import streamlit as st
import os
import tempfile
from rag_chain import create_vectorstore_from_pdf, get_rag_chain

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("RAG チャットボット 🤖")

# サイドバー: 設定とファイルアップロード
with st.sidebar:
    st.header("設定")
    # APIキーが環境変数にない場合のみ入力欄を表示、あるいは上書き用
    api_key_input = st.text_input("OpenAI API Key", type="password", help="設定済みの場合は空欄でOK")
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
    
    st.header("ドキュメント")
    uploaded_files = st.file_uploader("PDFをアップロード", type=["pdf"], accept_multiple_files=True)
    
    if st.button("クリア"):
        st.session_state.messages = []
        st.session_state.rag_chain = None
        st.rerun()

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# ファイルがアップロードされたらベクトルストアを作成
if uploaded_files and st.session_state.rag_chain is None:
    with st.spinner("ドキュメントを処理中..."):
        # アップロードファイルの保存先ディレクトリ
        upload_dir = "uploaded_docs"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
            
        file_paths = []
        try:
            # 各ファイルを保存
            for uploaded_file in uploaded_files:
                file_path = os.path.join(upload_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)
            
            # ベクトルストアとチェーンの作成
            vectorstore = create_vectorstore_from_pdf(file_paths)
            st.session_state.rag_chain = get_rag_chain(vectorstore)
            st.success(f"{len(uploaded_files)}個のファイルを読み込みました！質問してください。")
            
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

# チャット履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザー入力
if prompt := st.chat_input("質問を入力..."):
    # ユーザーのメッセージを表示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # アシスタントの応答
    if st.session_state.rag_chain:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("考え中..."):
                try:
                    response = st.session_state.rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    message_placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # 参照ソースの表示
                    if "context" in response and response["context"]:
                        context_docs = response["context"]
                        # ソースごとにページ番号を集約
                        sources_map = {}
                        for doc in context_docs:
                            source_path = doc.metadata.get("source", "")
                            page_num = doc.metadata.get("page", 0) + 1  # 0始まりなので+1
                            if source_path:
                                if source_path not in sources_map:
                                    sources_map[source_path] = set()
                                sources_map[source_path].add(page_num)
                        
                        if sources_map:
                            st.markdown("---")
                            st.markdown("##### 📚 参照ソース")
                            for source_path, pages in sources_map.items():
                                file_name = os.path.basename(source_path)
                                pages_str = ", ".join(map(str, sorted(pages)))
                                
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.caption(f"📄 **{file_name}** (p.{pages_str})")
                                with col2:
                                    # ボタンのキーを一意にするためにファイル名を使用
                                    if st.button("開く", key=f"open_{file_name}"):
                                        try:
                                            os.startfile(source_path)
                                        except Exception as e:
                                            st.error(f"ファイルを開けませんでした: {e}")
                                            
                except Exception as e:
                    message_placeholder.error(f"エラー: {e}")
    else:
        with st.chat_message("assistant"):
            st.warning("まずは左側のサイドバーからPDFをアップロードしてください。")
            st.session_state.messages.append({"role": "assistant", "content": "まずは左側のサイドバーからPDFをアップロードしてください。"})
