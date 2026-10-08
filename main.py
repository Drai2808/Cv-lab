import sys
from config import get_args
from ingestion import load_and_split
from features import extract_batch
from modeling import train_model, predict
from evaluation import Dashboard
def main():
    args = get_args()
    X_train, X_test, y_train, y_test = load_and_split(args.dataset, args.test_size)
    X_train_feats = extract_batch(X_train, args.canny_low, args.canny_high)
    X_test_feats = extract_batch(X_test, args.canny_low, args.canny_high)
    model = train_model(X_train_feats, y_train)
    y_pred = predict(model, X_test_feats)
    dashboard = Dashboard()
    dashboard.print_metrics(y_test, y_pred, args.output_img)
if __name__ == "__main__":
    main()