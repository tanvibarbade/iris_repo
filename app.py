import streamlit as st
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = sns.load_dataset('iris')


model = joblib.load("iris_model.pkl")
page = st.sidebar.selectbox("Select Option Here",["Home","EDA","Model"])

if page == "Home":
    st.title("**Iris Model Deployment⚙️**")
    st.markdown("""
### About the Model:
This is a Logistic Regression model trained on the **Iris Dataset**. The Iris dataset consists of 150 samples from three species of Iris flowers (Setosa, Versicolor, Virginica). 
The model predicts the species of the Iris flower based on four features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The model was trained using a train-test split with 80% data used for training and 20% data used for testing.

### Accuracy:
The model has been evaluated on the test data and shows an accuracy of:
""")

elif page == "EDA":
    st.write("EDA page")
    st.write(data.head())
    st.write('shape of data is',data.shape)
    st.markdown("""To check is there any missing value:""")
    st.write(data.isnull().sum())
    st.markdown("""Description about data:""")
    st.write(data.describe())
    
    st.markdown("""Iris Dataset Boxplots 📊""")
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(13, 5))
    for i, j in enumerate(numeric_data.columns, 1):
        plt.subplot(2, 3, i)
        plt.boxplot(numeric_data[j])
        plt.title(j)
    plt.tight_layout()
    st.pyplot(plt)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Distribution of Iris Features')
    sns.histplot(data['sepal_length'], bins=10, ax=axes[0, 0], kde=True)
    axes[0, 0].set_title('Sepal Length')
    axes[0, 0].set_xlabel("")

    sns.histplot(data['sepal_width'],bins=10,ax=axes[0,1],kde=True)
    axes[0,1].set_title('sepal_width')
    axes[0, 1].set_xlabel("")

    sns.histplot(data['petal_length'],bins=10,ax=axes[1,0],kde=True)
    axes[1,0].set_xlabel("petal_length")

    sns.histplot(data['petal_width'],bins=10,ax=axes[1,1],kde=True)
    axes[1,1].set_xlabel("petal_width")
    st.pyplot(fig)

else:
    st.title("**Prediction Page🔎**")
    sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.8)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.0)
    petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.35)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.3)

    if st.button("Predict"):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)
        
        st.write(f"Predicted Species: **{prediction}**")


