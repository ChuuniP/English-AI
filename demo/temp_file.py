import os
import random
import re
import requests
import urllib.parse

# Thư mục đích của bạn
target_dir = r"E:\Antigravity\English AI\datasets\sentences"
os.makedirs(target_dir, exist_ok=True)

# Danh sách kho câu mẫu phong phú
base_sentences = [
    "How are you doing today?", "I would like a cup of coffee.", "Could you please help me?",
    "Where is the nearest train station?", "What time does the store open?", "The weather is really nice today.",
    "I am learning English every day.", "Can I have the check please?", "Nice to meet you.",
    "What is your favorite food?", "I need to go to the supermarket.", "How much does this item cost?",
    "Do you speak English?", "I am looking for a book.", "Have a wonderful day ahead.",
    "That sounds like a great idea.", "I do not understand what you mean.", "Please turn off the lights.",
    "Where can I find a taxi?", "I am so happy to see you.", "What are your plans for the weekend?",
    "Could you speak a little slower?", "This meal is delicious.", "I have a question for you.",
    "Everything is going to be fine.", "What is the capital of France?", "I enjoy listening to music.",
    "She works at a technology company.", "He is reading an interesting book.", "We should take a short break.",
    "Can you give me a hand?", "I lost my key this morning.", "The train will arrive shortly.",
    "It is time to go to bed.", "Thank you very much for your time.", "I am feeling a bit tired.",
    "What is your phone number?", "Let us go for a walk in the park.", "Do you need any help?",
    "I agree with your opinion.", "Sorry for being late.", "What kind of movies do you like?",
    "I love eating fresh apples.", "Where did you buy this shirt?", "It is starting to rain.",
    "Please close the window.", "I need a glass of water.", "Who is your favorite artist?",
    "This problem is very simple.", "We need to work together.", "I hope to see you again soon.",
    "What is the meaning of this word?", "She speaks three languages fluently.", "He lives in a big city.",
    "Can you call me back later?", "I am busy with my homework.", "The water is too cold.",
    "They are playing football outside.", "I want to visit Japan next year.", "This is my favorite song.",
    "Do you have any suggestions?", "I need to practice speaking English.", "Where are you going right now?",
    "The restaurant is open now.", "Can I try this on?", "I need to buy some bread.",
    "The sun rises in the east.", "She is writing a letter.", "He likes to play basketball.",
    "We are planning a trip.", "It is a very beautiful day.", "I need to charge my phone.",
    "Where is the bathroom?", "How long does it take?", "I am waiting for a friend.",
    "What are you doing here?", "This is a great opportunity.", "I am sorry to hear that.",
    "Can you repeat that sentence?", "I like reading historical novels.", "She works as a doctor.",
    "He drives a red car.", "We are learning how to code.", "It is too hot today.",
    "I am looking for my glasses.", "Where did you go yesterday?", "What is your dream job?",
    "I am proud of your work.", "Please sit down and relax.", "The food smells amazing.",
    "Can you show me the way?", "I forgot my password.", "We are ready to start.",
    "It takes twenty minutes.", "I am looking forward to it.", "She is a talented singer.",
    "He answers all the questions.", "They are watching a movie.", "Everything is working well.",
    "I am ready to learn more."
]

# Đảm bảo lấy ngẫu nhiên đúng 100 câu
sample_sentences = random.sample(base_sentences, min(100, len(base_sentences)))

print(f"🚀 Đang tải {len(sample_sentences)} file MP3 câu tiếng Anh ngẫu nhiên vào:\n{target_dir}\n")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

success_count = 0
for idx, sentence in enumerate(sample_sentences, 1):
    encoded_text = urllib.parse.quote(sentence)
    tts_url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={encoded_text}&tl=en&client=tw-ob"

    try:
        response = requests.get(tts_url, headers=headers, timeout=10)
        if response.status_code == 200:
            # Tạo tên file từ câu
            clean_name = re.sub(r'[^\w\s]', '', sentence).strip().lower().replace(" ", "_")
            file_path = os.path.join(target_dir, f"{idx:03d}_{clean_name}.mp3")

            with open(file_path, "wb") as f:
                f.write(response.content)

            success_count += 1
            print(f"[{idx}/100] Đã tải: '{sentence}'")
        else:
            print(f"[{idx}/100] Lỗi HTTP {response.status_code} cho câu: '{sentence}'")
    except Exception as e:
        print(f"[{idx}/100] Lỗi khi tải câu '{sentence}': {e}")

print(f"\n🎉 Hoàn thành! Đã lưu thành công {success_count}/100 file MP3 vào {target_dir}")