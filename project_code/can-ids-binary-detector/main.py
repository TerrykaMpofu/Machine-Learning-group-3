from app.predict import predict_from_csv

if __name__ == "__main__":
    input_file = "sample_data/sample_input.csv"
    output_file = "sample_data/predictions_output.csv"
    predict_from_csv(input_file, output_file)
    print(f"Predictions saved to {output_file}")