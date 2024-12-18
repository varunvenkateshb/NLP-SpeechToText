import flet as ft
from gtts import gTTS
import os
import pygame
import speech_recognition as sr
from deep_translator import GoogleTranslator
import time
import nltk
from nltk.tokenize import word_tokenize
from pydub import AudioSegment
import noisereduce as nr
import librosa
import soundfile as sf
from langdetect import detect  # Language detection
import sqlite3
from history import setup_history_database, add_translation_to_history, get_user_history, group_history_by_date


# Database setup for users and history
def setup_database():
    conn = sqlite3.connect("users_and_history.db")
    c = conn.cursor()

    # Create users table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    )
    ''')

    # Create history table with all necessary columns
    c.execute('''
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        input_text TEXT NOT NULL,
        translated_text TEXT NOT NULL,
        source_lang TEXT NOT NULL,
        target_lang TEXT NOT NULL,
        conversion_type TEXT NOT NULL,
        user_id INTEGER NOT NULL,
        date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    ''')

    conn.commit()
    conn.close()
    
def upgrade_history_table():
    conn = sqlite3.connect("users_and_history.db")
    c = conn.cursor()

    # Check if the history table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='history'")
    if c.fetchone() is not None:
        # Create a new temporary table with the correct schema
        c.execute('''
        CREATE TABLE IF NOT EXISTS history_temp (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT NOT NULL,
            translated_text TEXT NOT NULL,
            source_lang TEXT NOT NULL,
            target_lang TEXT NOT NULL,
            conversion_type TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Copy data from old history table to new one (excluding missing columns)
        c.execute('''
        INSERT INTO history_temp (input_text, translated_text, source_lang, target_lang, user_id)
        SELECT input_text, translated_text, source_lang, target_lang, user_id FROM history
        ''')

        # Drop old history table
        c.execute("DROP TABLE history")

        # Rename new table to original name
        c.execute("ALTER TABLE history_temp RENAME TO history")

    conn.commit()
    conn.close()

def get_user_id(user_name):
    conn = sqlite3.connect("users_and_history.db")
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE name = ?", (user_name,))
    user_id = c.fetchone()
    conn.close()
    
    return user_id[0] if user_id else None  # Return None if no ID found


# Function to add a user to the database
def add_user(name, email, password):
    try:
        conn = sqlite3.connect("users_and_history.db")
        c = conn.cursor()
        c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:  # Email already exists
        return False

# Function to validate login credentials
def validate_login(email, password):
    conn = sqlite3.connect("users_and_history.db")
    c = conn.cursor()
    c.execute("SELECT name, password FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()
    
    if user and user[1] == password:
        return user[0]  # Return the user's name if login is successful
    else:
        return None

# Function to add a translation to history
def add_translation(input_text, translated_text, source_lang, target_lang, conversion_type, user_id):
    conn = sqlite3.connect("users_and_history.db")
    c = conn.cursor()
    c.execute("INSERT INTO history (input_text, translated_text, source_lang, target_lang, conversion_type, user_id) VALUES (?, ?, ?, ?, ?, ?)",
              (input_text, translated_text, source_lang, target_lang, conversion_type, user_id))
    conn.commit()
    conn.close()

# Fetch translation history
def get_history():
    conn = sqlite3.connect("users_and_history.db")
    c = conn.cursor()
    c.execute("SELECT * FROM history ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows

# Initialize recognizer and pygame for playing audio
recognizer = sr.Recognizer()
pygame.mixer.init()

nltk.download('punkt', quiet=True)

# Function to play audio
def play_audio(audio_file):
    try:
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
        pygame.mixer.music.stop()
    except Exception as ex:
        print(f"Audio playback error: {ex}")

# Function to safely remove a file
def safe_remove(file_path):
    try:
        os.remove(file_path)
    except PermissionError:
        time.sleep(0.1)

# Function to reduce noise in an audio file
def reduce_noise(file_path, output_path):
    print("Reducing noise in audio...")
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=sample_rate)
        sf.write(output_path, reduced_noise_audio, sample_rate)
        print("Noise reduction complete.")
        return output_path
    except Exception as ex:
        print(f"Error reducing noise: {ex}")
        return file_path

# Function to record audio from the microphone and save to a file
def record_audio_to_file(page, output_text, filename="train_announcement.wav", mic_index=None):
    try:
        mic = sr.Microphone(device_index=mic_index) if mic_index is not None else sr.Microphone()
        with mic as source:
            output_text.value = "Adjusting for ambient noise, please wait..."
            page.update()  # Update the output field in real-time
            recognizer.adjust_for_ambient_noise(source, duration=5)
            output_text.value = f"Energy threshold: {recognizer.energy_threshold}\nListening... Please speak."
            page.update()
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=60)
            with open(filename, "wb") as f:
                f.write(audio.get_wav_data())
            output_text.value = f"Audio saved to '{filename}'."
            return filename
    except sr.WaitTimeoutError:
        output_text.value = "Listening timed out while waiting for a phrase."
        return None
    except Exception as ex:
        output_text.value = f"Error while recording: {ex}"
        return None

# Function to transcribe the audio file with retries and multilingual support
def transcribe_audio_with_retries(audio_file_path, language=None, retries=3):
    best_transcription = ""
    best_token_count = 0

    for attempt in range(retries):
        print(f"Attempt {attempt + 1} to transcribe...")
        try:
            with sr.AudioFile(audio_file_path) as source:
                audio_data = recognizer.record(source)
                transcript = recognizer.recognize_google(audio_data, language=language) if language else recognizer.recognize_google(audio_data)
                print(f"Transcription result: {transcript}")  # Log the result for debugging

                # Calculate token count for accuracy (longer transcriptions usually more accurate)
                token_count = len(transcript.split())
                if token_count > best_token_count:
                    best_token_count = token_count
                    best_transcription = transcript

            if best_transcription:
                print(f"Best transcription so far: {best_transcription}")
            else:
                print(f"Attempt {attempt + 1} failed to improve transcription.")
        except sr.UnknownValueError:
            print("Could not understand audio.")
        except sr.RequestError as e:
            print(f"Google API error: {e}")
        except Exception as ex:
            print(f"Listening failed: {ex}")

    return best_transcription if best_transcription else "Could not understand audio after multiple attempts"

# Function to process audio with language detection and translation to native script
def process_audio_with_translation(page, audio_file_path, output_text, user_name):
    transcript = transcribe_audio_with_retries(audio_file_path)
    
    if "Could not understand audio" not in transcript:
        detected_lang = detect(transcript)
        output_text.value += f"\nDetected language: {detected_lang}"
        
        # Translate to English if detected language is not English
        if detected_lang != 'en':
            try:
                translated_transcript = GoogleTranslator(source='auto', target='en').translate(transcript)
                output_text.value += f"\nTranscription in English: {translated_transcript}"
                user_id = get_user_id(user_name)
                add_translation(transcript, translated_transcript, detected_lang, 'en', 'speech_to_text', user_id)
                return translated_transcript
            except Exception as e:
                output_text.value += f"\nTranslation error: {e}"
                return transcript
        return transcript
    return transcript

# Text-to-Speech conversion
def text_to_speech(text, lang):
    if text:
        try:
            pygame.mixer.music.stop()
            filename = f"text_{int(time.time())}.mp3"
            tts = gTTS(text=text, lang=lang)
            tts.save(filename)
            play_audio(filename)
            safe_remove(filename)
        except Exception as ex:
            print(f"Text-to-speech error: {ex}")

# Function to translate text and play in the translated language
def translate_and_speak_text(tts_textbox, trans_langbox, tts_translated_text, page,user_name):
    text = tts_textbox.value
    target_lang = trans_langbox.value
    if text and target_lang:
        try:
            translated_text = GoogleTranslator(source='auto', target=target_lang).translate(text)
            tts_translated_text.value = f"Translated Text: {translated_text}"
            page.update()  # Update the page to reflect the new translated text
            text_to_speech(translated_text, target_lang)
            user_id = get_user_id(user_name)  # Retrieve user ID
            add_translation(text, translated_text, 'en', target_lang, 'text_to_speech', user_id)
        except Exception as ex:
            tts_translated_text.value = f"Translation error: {ex}"
            page.update()

# Main function for Flet app
def main(page: ft.Page):
    setup_database()  # Ensure database is set up correctly
    upgrade_history_table()
    user_name = None  # Placeholder for the logged-in user's name
    right_panel_content = ft.Container()  # Placeholder for dynamic right panel content

    # Function to switch the right panel content based on feature selection
    def switch_right_panel(view_name):
        if view_name == "speech_to_text":
            right_panel_content.content = speech_to_text_view()
        elif view_name == "text_to_speech":
            right_panel_content.content = text_to_speech_view()
        elif view_name == "history":
            right_panel_content.content = history_view()
        else:
            right_panel_content.content = default_home_view()

        right_panel_content.update()

    # Function to show a SnackBar
    def show_snackbar(message, color):
        snack_bar = ft.SnackBar(ft.Text(message), bgcolor=color)
        page.overlay.append(snack_bar)
        snack_bar.open = True
        page.update()

    # Login view
    def login_view():
        def login(e):
            nonlocal user_name
            email = email_field.value
            password = password_field.value
            user = validate_login(email, password)  # Validate from database
            if user:
                user_name = user  # Set logged-in user's name
                switch_view("home")
            else:
                show_snackbar("Invalid email or password", ft.colors.RED)

        def redirect_to_signup(e):
            switch_view("signup")

        email_field = ft.TextField(label="Email", width=300)
        password_field = ft.TextField(label="Password", width=300, password=True)
        login_button = ft.ElevatedButton(text="Login", on_click=login)

        # "Sign Up" button for users who don't have an account
        signup_redirect_button = ft.TextButton("Don't have an account? Sign up here!", on_click=redirect_to_signup)

        return ft.View(
            controls=[
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text("Login", size=40, weight="bold"),
                            email_field,
                            password_field,
                            login_button,
                            signup_redirect_button,
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    alignment=ft.alignment.center,
                    expand=True
                )
            ]
        )

    # Signup view
    def signup_view():
        def signup(e):
            name = name_field.value
            email = email_field.value
            password = password_field.value
            if add_user(name, email, password):  # Check if user is added successfully
                show_snackbar("Sign up successful! You can now log in.", ft.colors.GREEN)
                switch_view("login")  # Redirect to login after successful signup
            else:
                show_snackbar("Email already exists!", ft.colors.RED)

        name_field = ft.TextField(label="Name", width=300)
        email_field = ft.TextField(label="Email", width=300)
        password_field = ft.TextField(label="Password", width=300, password=True)
        signup_button = ft.ElevatedButton(text="Sign Up", on_click=signup)

        return ft.View(
            controls=[
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text("Sign Up", size=40, weight="bold"),
                            name_field,
                            email_field,
                            password_field,
                            signup_button,
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    alignment=ft.alignment.center,
                    expand=True
                )
            ]
        )

    # Home view after successful login
    def home_view():
        def select_feature(e):
            feature = e.control.data
            switch_right_panel(feature)  # Switch content on the right side

        # Fixed left side panel
        side_panel = ft.Container(
            width=250,
            bgcolor=ft.colors.BLUE_GREY_900,
            padding=20,
            border_radius=ft.border_radius.all(30),
            content=ft.Column(
                [
                    ft.CircleAvatar(
                        content=ft.Text("OM", color=ft.colors.WHITE),
                        radius=40,
                        bgcolor=ft.colors.PINK_400
                    ),
                    ft.Text(f"{user_name}", size=24, color=ft.colors.WHITE, weight="bold"),
                    ft.TextButton("Speech-to-Text", data="speech_to_text", on_click=select_feature, style=ft.ButtonStyle(color=ft.colors.WHITE)),
                    ft.TextButton("Text-to-Speech", data="text_to_speech", on_click=select_feature, style=ft.ButtonStyle(color=ft.colors.WHITE)),
                    ft.TextButton("History", data="history", on_click=select_feature, style=ft.ButtonStyle(color=ft.colors.WHITE)),
                    ft.TextButton("Log out", on_click=lambda _: switch_view("login"), style=ft.ButtonStyle(color=ft.colors.RED))
                ],
                spacing=20,
                alignment=ft.MainAxisAlignment.START,
                horizontal_alignment=ft.CrossAxisAlignment.START
            ),
        )

        # Right side panel (dynamic content based on feature selection)
        return ft.View(
            controls=[
                ft.Row(
                    [
                        side_panel,
                        ft.VerticalDivider(width=1, color=ft.colors.BLACK12),
                        right_panel_content
                    ],
                    expand=True
                )
            ]
        )

    # Speech-to-Text feature view with additional functionalities
    def speech_to_text_view():
        stt_textbox = ft.TextField(label="Recognized Speech", multiline=True, width=500, height=150)
        stt_translation = ft.TextField(label="Translated Speech", multiline=True, width=500, height=150)
        trans_langbox = ft.TextField(label="Enter target language code", width=250)
        output_text = ft.Text()

        is_listening = False
        def start_listening(e):
            nonlocal is_listening
            if not is_listening:
                audio_file_path = record_audio_to_file(page, output_text)  # Start recording
                page.update()
                is_listening = True
            # Placeholder: Add actual code to start recording audio

        def stop_listening(e):
            nonlocal is_listening
            if is_listening:
                output_text.value = "Processing..."
                page.update()
                audio_file_path = "train_announcement.wav"
                if os.path.exists(audio_file_path):
                    transcript = process_audio_with_translation(page, audio_file_path, output_text, user_name)
                    stt_textbox.value = transcript
                    page.update()
                else:
                    output_text.value = f"Audio file '{audio_file_path}' not found."
                is_listening = False
            page.update()

        def translate_speech(e):
            translate_and_speak_text(stt_textbox, trans_langbox, stt_translation, page, user_name)
            page.update()

        def clear_speech_to_text(e):
            stt_textbox.value = ""
            stt_translation.value = ""
            output_text.value = ""
            trans_langbox.value = ""
            page.update()
        return ft.Container(
            content=ft.Column(
                [
                     ft.Text("Speech-To-Text Converter", size=25, weight="bold", color=ft.colors.RED),
                            ft.ElevatedButton(text="Start Listening", on_click=start_listening),
                            ft.ElevatedButton(text="Stop Listening and Process", on_click=stop_listening),
                            stt_textbox,
                            trans_langbox,
                            ft.ElevatedButton(text="Translate Speech", on_click=translate_speech),
                            stt_translation,
                            ft.ElevatedButton(text="Clear", on_click=clear_speech_to_text),
                            output_text,
                ],
                expand=True,
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=20
        )
    page.update()

    # Text-to-Speech feature view with functionalities
    def text_to_speech_view():
        tts_textbox = ft.TextField(label="Text to Speak", multiline=True, width=500, height=150)
        tts_translated_text = ft.TextField(label="Translated Text", multiline=True, width=500, height=150)
        trans_langbox = ft.TextField(label="Enter target language code", width=250)

        def speak_text(e):
            text_to_speech(tts_textbox.value, "en")

        def clear_tts(e):
            tts_textbox.value = ""
            tts_translated_text.value = ""
            trans_langbox.value = ""
            page.update()

        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Text-To-Speech Converter", size=25, weight="bold", color=ft.colors.RED),
                            tts_textbox,
                            ft.ElevatedButton(text="Speak", on_click=speak_text),
                            trans_langbox,
                            ft.ElevatedButton(text="Translate and Speak", on_click=lambda e: translate_and_speak_text(tts_textbox, trans_langbox, tts_translated_text, page, user_name)),
                            tts_translated_text,
                            ft.ElevatedButton(text="Clear", on_click=clear_tts),

                ],
                expand=True,
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=20
        )
    page.update()

    # History feature view
    def history_view():
        user_id = get_user_id(user_name)  # Retrieve current user's ID
        history_list = get_user_history(user_id)
        content = [ft.Text("Translation History", size=24, weight="bold")]

        from collections import defaultdict
        grouped_history = defaultdict(list)

        for item in history_list:
            grouped_history[item[7]].append(f"{item[1]} -> {item[2]} ({item[3]} to {item[4]}) [{item[5]}]")  # item[7] is date

        for date, entries in grouped_history.items():
            content.append(ft.Text(f"Date: {date}"))
            for entry in entries:
                content.append(ft.Text(entry))

        return ft.Container(
            content=ft.Column(
                controls=content,
            ),
            expand=True,
            padding=20
        )

    # Default home view
    def default_home_view():
        return ft.Container(
            content=ft.Text("Welcome to the Speech and Translation App!", size=24, weight="bold"),
            padding=20
        )

    # Function to switch between different views (login, signup, home)
    def switch_view(view_name):
        if view_name == "login":
            page.views.clear()
            page.views.append(login_view())
        elif view_name == "signup":
            page.views.clear()
            page.views.append(signup_view())
        elif view_name == "home":
            page.views.clear()
            page.views.append(home_view())
        page.update()

    # Start with the login view
    switch_view("login")
    setup_database()

ft.app(target=main)
