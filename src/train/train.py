import torch
import pickle

from sklearn.model_selection import train_test_split

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.model.model import DepressionModel


# For reproducibility
torch.manual_seed(42)


def train_model():

    # -----------------------------
    # 1. Load + preprocess data
    # -----------------------------
    df = load_data()
    df_clean = preprocess_data(df)

    # -----------------------------
    # 2. Split features & target
    # -----------------------------
    X = df_clean.drop(columns=['Depression'])
    y = df_clean['Depression']

    # -----------------------------
    # 3. Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🔥 Save feature columns (for Streamlit app)
    feature_columns = X_train.columns
    with open("feature_columns.pkl", "wb") as f:
        pickle.dump(feature_columns, f)

    print("Feature columns saved!")

    # -----------------------------
    # 4. Convert to tensors
    # -----------------------------
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor  = torch.tensor(X_test.values, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor  = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # -----------------------------
    # 5. Model setup
    # -----------------------------
    input_size = X_train.shape[1]
    print("Input size:", input_size)

    model = DepressionModel(input_size)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # -----------------------------
    # 6. Training loop
    # -----------------------------
    epochs = 150

    for epoch in range(epochs):

        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f}")

    # -----------------------------
    # 7. Evaluation
    # -----------------------------
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        y_pred = (y_pred > 0.5).float()

    accuracy = (y_pred == y_test_tensor).float().mean()
    print("Test Accuracy:", accuracy.item())

    # -----------------------------
    # 8. Save model
    # -----------------------------
    torch.save(model.state_dict(), "model.pth")
    print("Model saved!")


if __name__ == "__main__":
    train_model()