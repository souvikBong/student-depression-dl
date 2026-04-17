import torch
from sklearn.model_selection import train_test_split

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.model.model import DepressionModel


def train_model():

    # Load + preprocess
    df = load_data()
    df_clean = preprocess_data(df)

    # Split
    X = df_clean.drop(columns=['Depression'])
    y = df_clean['Depression']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor  = torch.tensor(X_test.values, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor  = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Model
    input_size = X_train.shape[1]
    model = DepressionModel(input_size)

    # Loss & optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 50

    for epoch in range(epochs):

        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Evaluation
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        y_pred = (y_pred > 0.5).float()

    accuracy = (y_pred == y_test_tensor).float().mean()
    print("Test Accuracy:", accuracy.item())


if __name__ == "__main__":
    train_model()
