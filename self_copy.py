import os

def main():
    # Read own source code from disk
    source_path = os.path.abspath(__file__)
    with open(source_path, "r", encoding="utf-8") as f:
        source_text = f.read()

    # Reserve a memory buffer and copy source into it
    buffer = bytearray(source_text.encode("utf-8"))

    print(f"Source file  : {source_path}")
    print(f"Source size  : {len(source_text)} characters")
    print(f"Buffer size  : {len(buffer)} bytes")

    # Verify the copy is identical
    recovered = buffer.decode("utf-8")
    if recovered == source_text:
        print("Verification : OK — buffer contents match the source file exactly")
    else:
        print("Verification : FAILED — mismatch detected")

    # Print the buffer contents
    print("\n--- Buffer contents ---")
    print(recovered)
    print("--- End of buffer ---")

if __name__ == "__main__":
    main()
