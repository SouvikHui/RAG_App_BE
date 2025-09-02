import os
from typing import List 
# from pytube import YouTube
from pathlib import Path
import tempfile
from langchain_core.documents import Document
from groq import Groq
from yt_dlp import YoutubeDL

from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY')) 

def download_youtube_audio(youtube_url: str) :
    try:
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(temp_dir.name)
        # print(temp_dir,temp_path)
        ytdl_out = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3', # Can choose mp3 or another format
            'preferredquality': '192', # Audio quality
                }],
        'outtmpl': str(temp_path / '%(title)s.%(ext)s'), # Output template for filename
        'writedescription': True, # Write video description
        'writeinfojson': True, # Write video metadata to a json file
                }
        with YoutubeDL(ytdl_out) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            # yt-dlp downloads the file, we need to find the downloaded file path
            # The output template should give us the file name
            # Find the downloaded file in the temp directory
            audio_files = list(temp_path.glob('*.mp3')) # Adjust extension based on preferredcodec
            if not audio_files:
                raise FileNotFoundError("Audio file not found after download.")
            audio_path = audio_files[0] # Assuming only one audio file is downloaded

            # Extract metadata from the info_dict
            metadata = {
                "source": youtube_url,
                "title": info_dict.get('title'),
                "author": info_dict.get('channel'), # Use 'channel' for author in yt-dlp
                "length_sec": info_dict.get('duration'),
                "description": info_dict.get('description')
            }
        return audio_path, metadata, temp_dir # Return temp_dir object for cleanup
    except Exception as e:
        print(f"Error downloading or processing video: {e}")
        # Clean up the temporary directory if an error occurs
        if temp_dir:  # Clean up only if temp_dir was successfully created
            temp_dir.cleanup()
        raise # Re-raise the exception after cleanup

def transcribe_with_groq(audio_path: Path) -> str:
    with open(audio_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
                            file=file,
                            model="whisper-large-v3-turbo",
                            prompt="Transcribe the audio file. Don't make up new things",
                            language="en",
                            response_format="json",
                            temperature=0.0,
                        )
    return transcription.text

def process_youtube_upload(youtube_url: str) -> List[Document]:
    docs=[]
    tmp_dir = None
    try:
        audio_path, metadata, tmp_dir = download_youtube_audio(youtube_url)
        transcript = transcribe_with_groq(audio_path)
        docs.append(Document(page_content=transcript,metadata=metadata))
    finally:
        if tmp_dir: # Clean up only if temp_dir was successfully created
            tmp_dir.cleanup()
    return docs

def process_audio_upload(uploaded_file) -> List[Document]:
    docs=[]
    temp_dir_path = tempfile.TemporaryDirectory()
    try:
        filename = uploaded_file.filename
        file_data = uploaded_file.file.read()
        file_path = Path(temp_dir_path.name) / filename
        with open(file_path, "wb") as f:
            f.write(file_data)
        transcript = transcribe_with_groq(file_path)
        docs.append(Document(page_content=transcript, metadata={"source": filename}))
    finally:
        if temp_dir_path:
            temp_dir_path.cleanup() 
    return docs