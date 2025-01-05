# ChatGPT did this

import openai
import pyttsx3
import pyaudio
import wave

# Audio recording configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
OUTPUT_FILENAME = "input_audio.wav"

def init_tts_engine():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speech rate
    return engine

def record_audio():
    """Record audio from the microphone and save it as a .wav file."""
    print("Recording... Speak now!")
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the audio file
    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    return OUTPUT_FILENAME

def query_gpt4o(audio_file_path):
    """Send audio to GPT-4o for transcription and response."""
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = openai.Audio.transcribe(
                model="gpt-4o",
                file=audio_file,
                prompt=(
                    "You are a language learning assistant. "
                    "The user is learning Japanese and their native language is English. "
                    "Help them practice Japanese, and if they ask questions or code-switch to English, answer appropriately."
                ),
            )
        return response.get("text", ""), response.get("language", "unknown")
    except Exception as e:
        print(f"Error querying GPT-4o: {e}")
        return None, None

def main():
    openai.api_key = "your_openai_api_key"  # Replace with your OpenAI API key
    tts_engine = init_tts_engine()

    print("Welcome to the Language Learning Conversation App!")
    print("You're learning Japanese, and your native language is English.")
    print("Feel free to speak in Japanese, English, or both.")

    while True:
        print("\nRecording audio... (type 'exit' to quit)")
        command = input("Press Enter to start recording or type 'exit': ").strip()

        if command.lower() == "exit":
            print("Goodbye! Keep practicing!")
            break

        audio_path = record_audio()
        print("Processing audio with GPT-4o...")

        transcription, detected_language = query_gpt4o(audio_path)

        if not transcription:
            print("Sorry, I couldn't process the audio. Try again.")
            continue

        print(f"Transcription: {transcription} (Detected language: {detected_language})")

        # Generate a contextual response based on the transcription
        response_prompt = (
            f"The user said: '{transcription}'. "
            "Provide a response in Japanese with explanations in English if needed. "
            "If the user switches to English, respond in English. "
            "Ensure the response helps the user learn Japanese."
        )

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": response_prompt},
                ],
            )
            assistant_response = response['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error querying GPT-4 for response: {e}")
            assistant_response = "Sorry, I couldn't process your request."

        print(f"Assistant: {assistant_response}")
        tts_engine.say(assistant_response)
        tts_engine.runAndWait()

if __name__ == "__main__":
    main()
