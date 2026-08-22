import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import gradio as gr

# 1) Read dataset directly from the same folder as app.py
df = pd.read_csv("german_credit.csv")

# 2) Use Credit Amount as the only independent variable
X = df[["Credit Amount"]]
y = df["Creditability"].astype(int)

# 3) Split data and train a simple logistic regression model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)

# 4) Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print(
    f"\nCredit Amount range in dataset: "
    f"{X['Credit Amount'].min():,.0f} to {X['Credit Amount'].max():,.0f}"
)

# 5) Prediction function
def predict_creditability(credit_amount):
    credit_amount = float(credit_amount)

    input_data = pd.DataFrame(
        {"Credit Amount": [credit_amount]}
    )

    prediction = int(model.predict(input_data)[0])
    probability = model.predict_proba(input_data)[0]

    label = (
        "1 - Creditworthy"
        if prediction == 1
        else "0 - Not Creditworthy"
    )

    return (
        label,
        f"Probability of Creditability = 1: {probability[1]:.2%}",
        f"Probability of Creditability = 0: {probability[0]:.2%}"
    )

# 6) Gradio web application
demo = gr.Interface(
    fn=predict_creditability,
    inputs=gr.Number(
        label="Credit Amount",
        value=2500
    ),
    outputs=[
        gr.Textbox(label="Predicted Creditability"),
        gr.Textbox(label="Probability: Creditability = 1"),
        gr.Textbox(label="Probability: Creditability = 0")
    ],
    title="Creditability Prediction",
    description=(
        "Enter a credit amount to predict Creditability (0 or 1). "
        "The model uses Credit Amount as the only independent variable "
        "and Logistic Regression for binary classification."
    ),
    examples=[
        [500],
        [1500],
        [3000],
        [5000],
        [10000]
    ]
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
