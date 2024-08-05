def print_hier_file_header(path, num_bytes=100):
    with open(path, 'rb') as f:
        header = f.read(num_bytes)
        print("Header bytes: ", header)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print header of hier file.")
    parser.add_argument("file_path", type=str, help="Path to the hier file.")
    args = parser.parse_args()

    print_hier_file_header(args.file_path)