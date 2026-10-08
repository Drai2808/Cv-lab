import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Normal vs Abnormal Image Classifier")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--test-size", type=float, default=0.4)
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--output-img", type=str, default="results.png")
    return parser.parse_args()