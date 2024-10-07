# test script to see if torch and cuda runs on the hpc

def main():
    import torch
    print(torch.__version__)
    print(f"cuda available: {torch.cuda.is_available()}")


if __name__ == '__main__':
    main()