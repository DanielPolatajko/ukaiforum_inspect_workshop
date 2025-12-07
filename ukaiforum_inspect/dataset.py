from inspect_ai.dataset import Sample, MemoryDataset


SECRET_KEY = "secret text"

def xor(s, t) -> bytes:
    """XOR two strings together with key repetition."""
    if isinstance(s, str):
        return "".join(chr(ord(s[i]) ^ ord(t[i % len(t)])) for i in range(len(s))).encode("utf8")
    else:
        return bytes([s[i] ^ t[i % len(t)] for i in range(len(s))])

message_1 = "Hello, world!"
message_2 = "UK AI Forum is awesome!"
message_3 = "Inspect is my favourite evals framework!"

messages = [
    (message_1, xor(message_1.encode(), SECRET_KEY.encode()).hex()),
    (message_2, xor(message_2.encode(), SECRET_KEY.encode()).hex()),
    (message_3, xor(message_3.encode(), SECRET_KEY.encode()).hex())
]

samples = [
    Sample(
        input=f"Message: {encrypted_message}\n\n Encryption key: {SECRET_KEY}\n\nCan you decrypt this message?",
        target=message
    )
    for (message, encrypted_message) in messages
]

decryption_dataset = MemoryDataset(samples)