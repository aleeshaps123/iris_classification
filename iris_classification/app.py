import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
data = iris.data
target = iris.target

# Split the data into features and target
X = data
y = target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train a TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, verbose=1)

# Evaluate the model
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
#st.write(f"Model Accuracy: {accuracy:.2f}")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        height: 100vh;
    }
    .sidebar .sidebar-content {
        padding-top: 20px;
        padding-right: 20px;
        padding-bottom: 20px;
        padding-left: 20px;
    }
    .prediction {
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .stApp {
        background: url('https://www.realestate.com.au/news-image/w_4704,h_3136/v1662074998/news-lifestyle-content-assets/wp-content/production/flower-trends-2019_3707481d2a8.jpg?_i=AA');
        background-size: cover;
    }
    .box {
        padding: 20px;
        margin: 20px 0;
        border: 1px solid #ddd;
        border-radius: 5px;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
    }
    </style>
    """, unsafe_allow_html=True)

# Main container
with st.container():
    st.title('Iris Flower Classification')

    # Sidebar sliders for user input
    st.sidebar.title("Iris Flower Classification")
    st.sidebar.write("### Predict Iris Flower Type")
    sepal_length = st.sidebar.slider('Sepal Length', float(X[:, 0].min()), float(X[:, 0].max()), step=0.1)
    sepal_width = st.sidebar.slider('Sepal Width', float(X[:, 1].min()), float(X[:, 1].max()), step=0.1)
    petal_length = st.sidebar.slider('Petal Length', float(X[:, 2].min()), float(X[:, 2].max()), step=0.1)
    petal_width = st.sidebar.slider('Petal Width', float(X[:, 3].min()), float(X[:, 3].max()), step=0.1)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    try:
        input_data_scaled = scaler.transform(input_data)  # Scale input data
    except Exception as e:
        st.error(f"Error in scaling input data: {e}")
        input_data_scaled = None

    if st.button('Predict'):
        if input_data_scaled is not None:
            try:
                # TensorFlow prediction
                tf_prediction = model.predict(input_data_scaled)
                tf_predicted_class = iris.target_names[np.argmax(tf_prediction)]
                st.markdown(f'<div class="box"><p class="prediction">Predicted Iris Flower Type: {tf_predicted_class}</p></div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error in prediction: {e}")
        else:
            st.error("Input data scaling failed. Check the input values.")
    else:
        st.markdown('<div class="box"><p>Click the button to predict the iris flower type.</p></div>', unsafe_allow_html=True)
