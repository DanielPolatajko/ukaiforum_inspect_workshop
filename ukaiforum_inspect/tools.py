from inspect_ai.tool import tool, Tool

@tool
def str_to_binary() -> Tool:
    async def execute(input: str) -> str:
        """
        Encode a string into a binary string.

        Args:
            input: The string to encode.

        Returns:
            The binary string.
        """
        return "".join(format(ord(char), "08b") for char in input)
    return execute

@tool
def hex_to_binary() -> Tool:
    async def execute(input: str) -> str:
        """
        Encode a hex string into a binary string.

        Args:
            input: The hex string to encode.

        Returns:
            The binary string.
        """
        input = input.replace(" ", "")
        return "".join(format(int(input[i:i+2], 16), "08b") for i in range(0, len(input), 2))
    return execute

@tool
def binary_decoding() -> Tool:
    async def execute(input: str) -> str:
        """
        Decode a binary string into a string.

        Args:
            input: The binary string to decode.

        Returns:
            The decoded string.
        """
        input = input.replace(" ", "")
        return "".join(chr(int(input[i:i+8], 2)) for i in range(0, len(input), 8))
    return execute

@tool
def xor_binary() -> Tool:
    async def execute(first: str, second: str) -> str:
        """
        Given two binary strings, XORs them byte-by-byte and returns the decoded text.
        If the strings are different lengths, the shorter one repeats to match the longer one.

        Args:
            first: The first binary string.
            second: The second binary string.

        Returns:
            The XOR result as a binary string.
        """
        first = first.replace(" ", "")
        second = second.replace(" ", "")

        result = []
        for i in range(0, len(first), 8):
            byte1 = int(first[i:i+8], 2)
            # Repeat the key if necessary
            key_index = (i // 8) % (len(second) // 8)
            byte2 = int(second[key_index*8:(key_index+1)*8], 2)
            result.append(format(byte1 ^ byte2, '08b'))

        return "".join(result)
    return execute