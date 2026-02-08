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
        tmp_paths = []
        try:
            # 各ファイルを一時ファイルとして保存
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_paths.append(tmp_file.name)
            
            # ベクトルストアとチェーンの作成
            # 修正されたcreate_vectorstore_from_pdfはパスのリストを受け取る
            vectorstore = create_vectorstore_from_pdf(tmp_paths)
            st.session_state.rag_chain = get_rag_chain(vectorstore)
            st.success(f"{len(uploaded_files)}個のファイルを読み込みました！質問してください。")
            
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
        finally:
            # 一時ファイルの削除
            for path in tmp_paths:
                if os.path.exists(path):
                    os.remove(path)

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
                except Exception as e:
                    message_placeholder.error(f"エラー: {e}")
    else:
        with st.chat_message("assistant"):
            st.warning("まずは左側のサイドバーからPDFをアップロードしてください。")
            st.session_state.messages.append({"role": "assistant", "content": "まずは左側のサイドバーからPDFをアップロードしてください。"})
