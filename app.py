import streamlit as st
import os 
import base64
import json
import requests
import time
import tempfile
from pathlib import Path

# --- NEW IMPORT: Google Text-to-Speech ---
from gtts import gTTS

# --- Handle Import Error for Mistral ---
try:
    from mistralai import Mistral
except ImportError:
    st.error("Mistral library not found. Please install it using: pip install mistralai")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="Image to Audio AI Agent", page_icon="🤖", layout="wide")

# --- Helper Function: Save File ---
def save_file(content, filename, folder_path, file_type="binary"):
    try:
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, filename)
        if file_type == "text":
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            with open(file_path, "wb") as f:
                f.write(content)
        return file_path, True
    except Exception as e:
        return str(e), False

# --- HYBRID AUDIO FUNCTION (OpenAI + Google Free Fallback) ---
def convert_text_to_audio_from_session(text, api_key, voice="alloy", output_folder=str(Path.home() / "Audio_Results"), source_label="Session_Text"):
    
    # 1. Try OpenAI First (High Quality)
    if api_key and api_key.strip():
        try:
            header = {
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json"
            }
            url = "https://api.openai.com/v1/audio/speech"
            data = {
                "model": "tts-1",
                "input": text,
                "voice": voice,
                "format": "mp3"
            }
            
            response = requests.post(url, headers=header, json=data)
            
            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    tmp_file.write(response.content)
                    return True, tmp_file.name, response.content
            # If we get here, OpenAI failed (likely 429 or 401), so we continue to Google...
        except Exception:
            pass # Fail silently and move to step 2

    # 2. Fallback to Free Google TTS (gTTS)
    try:
        # Check if we are switching because of a failure or just missing key
        if api_key:
            st.warning("⚠️ OpenAI Quota exceeded or Key invalid. Switching to Free Google TTS (Robotic Voice)...")
        else:
            st.info("ℹ️ No OpenAI Key provided. Using Free Google TTS...")

        # Create Google Audio
        tts = gTTS(text=text, lang='en')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            
            # We need to read the bytes back to return them compatible with the rest of the app
            with open(tmp_file.name, "rb") as f:
                audio_content = f.read()
                
            return True, tmp_file.name, audio_content
            
    except Exception as e:
        return False, f"Error (Both OpenAI and Google Failed): {str(e)}", None


# --- Initialize Session State ---
if "ocr_result" not in st.session_state:
    st.session_state.ocr_result = []
if "preview_src" not in st.session_state:
    st.session_state.preview_src = []
if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = []
if "audio_result" not in st.session_state:
    st.session_state.audio_result = ""
if "text_to_convert" not in st.session_state:
    st.session_state.text_to_convert = ""


# --- TABS ---
tab1, tab2 = st.tabs(["OCR Text Extraction", " Text to Audio Generation"])


# ==========================================
# TAB 1: OCR Text Extraction
# ==========================================
with tab1:
    st.title("OCR Application - Extract Text from Images")
    st.markdown("<h3 style='color:blue;'> Extract Text from Images using OCR Technology </h3>", unsafe_allow_html=True)

    # API Key Input
    api_key = st.text_input("Enter your Mistral AI API Key:", type="password")
    
    if not api_key:
        st.warning("Please enter your Mistral AI API Key to proceed.")

    output_folder = st.text_input(
        "Output Folder path for OCR results:",
        value=str(Path.home() / "OCR_Results"),
        help="Folder to save the extracted text files."
    )

    if os.path.exists(output_folder):
        st.success(f"Output folder exists at: {output_folder}") 
    else:
        st.info(f"Output folder does not exist. It will be created at: {output_folder}")  
        try:
            os.makedirs(output_folder, exist_ok=True)
            st.success(f"successfully created output folder at: {output_folder}")
        except Exception as e:
            st.error(f"unable to create folder: {str(e)}")

    col1, col2 = st.columns(2)

    with col1:
        file_type = st.radio("Select File Type:", ("Image", "PDF"), horizontal=True)

    with col2:
        source_type = st.radio("Select Source Type:", ("Upload", "URL"), horizontal=True)

    # Input based on source type
    input_url = ""
    uploaded_file = []

    if source_type == "URL":
        input_url = st.text_input("Enter one or multiple URL:(comma separated for multiple URLs)")
    else:
        uploaded_file = st.file_uploader(
            "Upload Image or PDF file",
            type=["png", "jpg", "jpeg", "pdf"],
            accept_multiple_files=True
        )

    # --- Process Button Logic ---
    if st.button("Process"):
        if not api_key:
            st.error("You must enter a Mistral API Key to process.")
            st.stop()

        if source_type == "URL" and not input_url.strip():
            st.warning("Please enter at least one URL to proceed.")
            st.stop()

        if source_type == "Upload" and not uploaded_file:
            st.warning("Please upload at least one file to proceed.")
            st.stop()

        # Initialize Mistral Client
        client = Mistral(api_key=api_key.strip())

        # Reset Session State
        st.session_state.ocr_result = []
        st.session_state.preview_src = []
        st.session_state.image_bytes = []
        
        # Determine sources
        if source_type == "URL":
            sources = input_url.split(",") 
        else:
            sources = uploaded_file

        for idx, source in enumerate(sources):
            document = None
            preview_src = None
            try:
                if source_type == "URL":
                    clean_url = source.strip()
                    preview_src = clean_url
                    if file_type == "PDF":
                        document = {"type": "pdf_url", "pdf_url": clean_url}
                    else:
                        document = {"type": "image_url", "image_url": clean_url}
                else:
                    file_bytes = source.read()
                    encoded_file = base64.b64encode(file_bytes).decode("utf-8")
                    mime_type = source.type if hasattr(source, "type") else "application/octet-stream"
                    document = {"type": "image_url", "image_url": f"data:{mime_type};base64,{encoded_file}"}
                    st.session_state.image_bytes.append(file_bytes)
                    preview_src = file_bytes

                with st.spinner(f"processing {source if source_type == 'URL' else source.name}..."):
                    ocr_response = client.ocr.process(
                        model="mistral-ocr-latest",
                        document=document,
                        include_image_base64=True
                    )
                    time.sleep(1)
                    pages = []
                    if hasattr(ocr_response, "pages"):
                        pages = ocr_response.pages
                    elif isinstance(ocr_response, list):
                        pages = ocr_response
                    
                    result_text = "\n\n".join(page.markdown for page in pages) or "No result found."
                    st.session_state.ocr_result.append(result_text)
                    st.session_state.preview_src.append(preview_src)

            except Exception as e:
                error_msg = f"Error Extracting result: {str(e)}"
                st.session_state.ocr_result.append(error_msg)
                st.session_state.preview_src.append(None)
                st.error(error_msg)


    # --- Display Results Section ---
    if st.session_state.ocr_result:
        for idx, result in enumerate(st.session_state.ocr_result):
            st.subheader(f"OCR Result for Source {idx+1}")
            d_col1, d_col2 = st.columns(2)
            
            with d_col1:
                st.subheader(f"Input Preview")
                preview_item = st.session_state.preview_src[idx]
                if preview_item:
                    if isinstance(preview_item, bytes):
                        st.image(preview_item, width=400)
                    elif isinstance(preview_item, str):
                        st.image(preview_item, width=400)
                else:
                    st.write("No preview available")

            with d_col2:
                st.subheader("OCR Result Text")
                edited_text = st.text_area("Extracted Text:", value=result, height=400, key=f"editor_{idx}")
                st.session_state.ocr_result[idx] = edited_text

                btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
                with btn_col1:
                    json_data = json.dumps({"ocr_result": edited_text}, indent=2, ensure_ascii=False)
                    b64_json = base64.b64encode(json_data.encode()).decode()
                    st.markdown(f'<a href="data:application/json;base64,{b64_json}" download="ocr_result_{idx+1}.json">Download JSON</a>', unsafe_allow_html=True)
                with btn_col2:
                    text_b64 = base64.b64encode(edited_text.encode()).decode()
                    st.markdown(f'<a href="data:text/plain;base64,{text_b64}" download="ocr_result_{idx+1}.txt">Download Text</a>', unsafe_allow_html=True)
                with btn_col3:
                    if st.button(f"Save JSON", key=f"save_json_{idx}"):
                        saved_path, success = save_file(json_data, f"output_{idx+1}.json", output_folder, file_type="text")
                        if success: st.success(f"Saved JSON")
                        else: st.error(f"Error saving JSON")
                with btn_col4:
                    if st.button(f"Save Text", key=f"save_text_{idx}"):
                        saved_path, success = save_file(edited_text, f"output_{idx+1}.txt", output_folder, file_type="text")
                        if success: st.success(f"Saved Text")
                        else: st.error(f"Error saving Text")

                if st.button(f"Convert this Result to Audio", key=f"convert_audio_{idx}"):
                    st.session_state["text_to_convert"] = edited_text
                    st.session_state["audio_output_folder"] = output_folder
                    st.session_state["audio_source_label"] = f"OCR_Result_Source_{idx+1}"
                    st.success("Text sent to Audio Tab! Click on the 'Text to Audio Generation' tab above.")


# ==========================================
# TAB 2: Text to Audio Generation
# ==========================================
with tab2:
    st.title("Text to Audio Generation")
    
    st.info("ℹ️ Note: If you do not have a paid OpenAI Key, leave the field below blank. The app will automatically use the free Google Voice.")
    openai_api_key = st.text_input("Enter your OpenAI API Key (Optional):", type="password", key="openai_api_key")

    default_audio_path = str(Path.home() / "Audio_Results")
    initial_folder_value = st.session_state.get("audio_output_folder", default_audio_path)
    
    audio_output_folder = st.text_input(
        "Output Folder path for Audio files:",
        value=initial_folder_value,
        help="Folder to save the generated audio files.",
        key="audio_output_folder_input"
    )

    if not os.path.exists(audio_output_folder):
        try: os.makedirs(audio_output_folder, exist_ok=True)
        except: pass

    col1, col2 = st.columns(2)
    with col1:
        text_input_method = st.radio("Select Text Input Method:", ("Manual Input", "From OCR Result", "Upload file"), horizontal=True, key="text_input_method")
    with col2:
        voice_choice = st.selectbox("Select Voice style (Only for OpenAI):", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"], index=0, key="voice_choice")

    text_for_audio = ""
    if text_input_method == "Manual Input":
        text_for_audio = st.text_area("Enter Text to Convert to Audio:", height=300, key="manual_text_input") 
    elif text_input_method == "From OCR Result":
        text_for_audio = st.session_state.get("text_to_convert", "")
        if text_for_audio:
            st.success("Text loaded from OCR Result.")
            text_for_audio = st.text_area("Text to Convert to Audio:", value=text_for_audio, height=300, key="ocr_text_input")
        else:
            st.warning("No OCR text found in session.")
    else:
        uploaded_text_file = st.file_uploader("Upload a Text File", type=["txt"], key="text_file_upload")
        if uploaded_text_file:
            text_for_audio = uploaded_text_file.read().decode("utf-8")
            st.success("Text file uploaded successfully.")
            text_for_audio = st.text_area("Text to Convert to Audio:", value=text_for_audio, height=300, key="file_text_input")

    if st.button("Generate Audio"):
        if not text_for_audio.strip():
            st.warning("Please provide some text.")
            st.stop()

        with st.spinner("Generating audio (checking availability)..."):
            # We pass the OpenAI key (even if empty) to the function
            success, audio_path_or_error, audio_content = convert_text_to_audio_from_session(
                text=text_for_audio,
                api_key=openai_api_key,
                voice=voice_choice,
                output_folder=audio_output_folder
            )
            
            if success:
                timestamp = int(time.time())
                audio_filename = f"audio_output_{timestamp}.mp3"
                saved_path, save_success = save_file(audio_content, audio_filename, audio_output_folder, file_type="binary")
                if save_success:
                    st.success(f"Audio file generated and saved at: {saved_path}")
                    st.audio(audio_content, format="audio/mp3")
                else:
                    st.error(f"Audio generated but failed to save: {saved_path}")
            else:
                st.error(f"Error generating audio: {audio_path_or_error}")

    st.markdown("---")
    st.subheader("Convert Text from Session to Audio (Shortcut)")
    if st.button("Convert Session Text to Audio Now"):
        if not text_for_audio:
            st.warning("No text found.")
        else:
            with st.spinner("Converting..."):
                success, path, content = convert_text_to_audio_from_session(text_for_audio, openai_api_key, voice_choice, audio_output_folder)
                if success:
                    st.success(f"Success! Saved at: {path}")
                    st.audio(content, format="audio/mp3")
                else:
                    st.error(f"Error: {path}")