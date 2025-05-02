import streamlit as st

# Title of the app
st.title("Plant Disease Detection Using CNN")

# Upload image section
uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("\nProcessing the image...")

    # Placeholder for model prediction (to be implemented)
    st.write("Prediction: [Model output here]")

# Footer
st.write("\nDeveloped using Streamlit")