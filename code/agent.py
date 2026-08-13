class SimpleChatSessionManager:
    def __init__(self):
        self.history = []

    def get_history_as_string(self):
        return "\n".join(self.history)

    def add_message(self, role, text):
        self.history.append(f"{role}: {text}")