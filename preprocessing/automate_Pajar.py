import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from pathlib import Path



def preprocess_data(input_path, output_path):
    """
    Fungsi untuk melakukan preprocessing dataset
    dan menyimpan hasilnya ke file baru.
    """

    # Load dataset
    df = pd.read_csv(input_path)

    # Hapus data duplikat
    df = df.drop_duplicates()

    # Pisahkan fitur dan target
    X = df.drop("depression_label", axis=1)
    y = df["depression_label"]

    # Encoding fitur kategorikal
    categorical_columns = [
        "gender",
        "platform_usage",
        "social_interaction_level"
    ]

    for col in categorical_columns:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])

    # Feature scaling
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Konversi kembali ke DataFrame
    processed_df = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    # Tambahkan target
    processed_df["depression_label"] = y.values

    # Simpan hasil preprocessing
    processed_df.to_csv(
        output_path,
        index=False
    )

    print(f"Dataset berhasil diproses dan disimpan di: {output_path}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent

    INPUT_PATH = BASE_DIR.parent / "dataset_raw" / "Teen_Mental_Health_Dataset.csv"
    OUTPUT_PATH = BASE_DIR.parent / "preprocessing" / "Teen_Mental_Health_Dataset_Processed.csv"

    print("Input :", INPUT_PATH)
    print("Output :", OUTPUT_PATH)