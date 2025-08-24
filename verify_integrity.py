"""Verify file integrity of EVER framework files"""
import hashlib
import os

def calculate_checksums(directory):
    """Calculate checksums for all Python files in directory"""
    checksums = {}
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'rb') as f:
                    content = f.read()
                    checksums[filepath] = hashlib.sha256(content).hexdigest()
    
    return checksums

def main():
    """Main function"""
    checksums = calculate_checksums('src')
    
    print("EVER Framework File Checksums:")
    for filepath, checksum in sorted(checksums.items()):
        print(f"{filepath}: {checksum}")
    
    # Save checksums to file
    with open('checksums.txt', 'w') as f:
        for filepath, checksum in sorted(checksums.items()):
            f.write(f"{filepath}: {checksum}\n")
    
    print("\nChecksums saved to checksums.txt")

if __name__ == "__main__":
    main()