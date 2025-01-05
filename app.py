import os
import openai
import whisper
import pyaudio
import wave
import time

###############################################################################
# Configuration
###############################################################################
# Replace with your own OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# Adjust the user's Japanese skill level: "beginner", "intermediate", or "advanced"
user_japanese_proficiency = "beginner"

###############################################################################
# Audio Recording Parameters
###############################################################################
CHUNK = 1024           # Number of frames per buffer
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000           # Whisper models often work best with 16 kHz audio
RECORD_SECONDS = 5     # Length of each audio capture in seconds
TEMP_WAV = "temp.wav"  # Temporary file to store audio

###############################################################################
# Initialize Whisper Model
###############################################################################
# You can choose different model sizes: "tiny", "base", "small", "medium", "large"
# Larger models are more accurate but use more resources.
print("Loading Whisper model (this can take some time)...")
model = whisper.load_model("small")  # e.g. "small", "medium", etc.

###############################################################################
# Helper Functions
###############################################################################
def build_system_prompt(proficiency_level="beginner"):
    """
    Build the system role instructions for GPT based on user proficiency level.
    """
    if proficiency_level == "beginner":
        level_instructions = (
            "Use simple vocabulary and short sentences. "
            "Provide brief explanations in English if the user seems confused."
        )
    elif proficiency_level == "intermediate":
        level_instructions = (
            "Use moderate-level vocabulary in Japanese, "
            "explain grammar points in English if asked, but keep the conversation primarily in Japanese."
        )
    elif proficiency_level == "advanced":
        level_instructions = (
            "Use natural, fluent Japanese with more advanced vocabulary. "
            "Only switch to English if the user specifically asks for an explanation."
        )
    else:
        level_instructions = (
            "Use Japanese appropriately. Switch to English for clarifications upon user request."
        )
    
    system_prompt = (
        "You are a Japanese language tutor. The user might speak in Japanese or English, "
        "and may switch languages mid-sentence. Continue responding mostly in Japanese if they are "
        "practicing, but provide English explanations when needed. "
        f"{level_instructions}"
    )
    return system_prompt


def get_llm_response(user_message, conversation_history, proficiency_level):
    """
    Sends the conversation (system + user) to the OpenAI LLM and returns the model's response.
    """
    # System prompt with language tutoring context
    system_prompt = build_system_prompt(proficiency_level)
    
    # Prepare the conversation for the ChatCompletion endpoint
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    # Add the existing conversation history
    messages.extend(conversation_history)
    # Add the newest user message
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4", etc.
            messages=messages,
            temperature=0.7
        )
        assistant_reply = response["choices"][0]["message"]["content"]
        return assistant_reply
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "I'm sorry, I couldn't process that. Please try again."


###############################################################################
# Main POC Loop
###############################################################################
def main():
    print("=== Japanese Language Learning App (POC with Whisper) ===")
    print("Speak into your microphone for about 5 seconds each time.")
    print("Say 'quit' in English to exit the loop.\n")
    
    conversation_history = []
    audio_interface = pyaudio.PyAudio()
    
    try:
        while True:
            # Step 1: Record audio from microphone
            print("Recording... (5 seconds)")
            stream = audio_interface.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            frames = []
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            
            # Step 2: Save recording to a WAV file
            wf = wave.open(TEMP_WAV, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio_interface.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            # Step 3: Transcribe with Whisper
            print("Transcribing with Whisper...")
            result = model.transcribe(TEMP_WAV)
            user_text = result["text"].strip()
            
            if not user_text:
                print("No speech detected. Try again.")
                continue
            
            print(f"You said: {user_text}")
            
            # Step 4: Check if user wants to quit
            if user_text.lower() == "quit":
                print("Exiting the app.")
                break
            
            # Step 5: Get GPT response
            conversation_history.append({"role": "user", "content": user_text})
            assistant_reply = get_llm_response(
                user_message=user_text,
                conversation_history=conversation_history,
                proficiency_level=user_japanese_proficiency
            )
            
            # Step 6: Print assistant's response
            print(f"\nGPT Tutor: {assistant_reply}\n")
            conversation_history.append({"role": "assistant", "content": assistant_reply})
            
            # Optionally, you can add TTS here to speak the response out loud
            
            # Small pause for user clarity
            time.sleep(1)
    
    finally:
        audio_interface.terminate()


if __name__ == "__main__":
    main()
