import streamlit as st
import keras
from PIL import Image, ImageOps
import numpy as np
import base64

new_title = '<p style="font-family:serif; color:black;border-radius:100px ;background-color: yellow;text-align: center; font-size: 42px;"><b>Tomato Disease Predictor</b></p>'
st.markdown(new_title, unsafe_allow_html=True)

new_title = '<p style="font-family:serif; color:black; border-radius:100px ;background-color: #aef32e;text-align: center;font-size: 23px;"><b>We Predict Tomato Disease from Its Leaves</b></p>'
st.markdown(new_title, unsafe_allow_html=True)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
     <style>
     .stApp {{
         background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
         background-size: cover
     }}
     </style>
     """,
        unsafe_allow_html=True
    )




add_bg_from_local("C:/Users/Yash/Tomato_disease_pred/1000_F_194267019_bsMbr1beD2xAREL5pXEB9VR3GBw7msnE.jpg")

def teachable_machine_classification(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 112, 112, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (112, 112)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32))

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction),np.max(prediction)# return position of the highest probability



uploaded_file = st.file_uploader("Choose an image ...", type=["jpg","png","jpeg"])
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption=None, use_column_width=True)
        new_title = '<p style="font-family:serif; color:black; font-size: 24px;text-align: center"><b>Classifying...</b></p>'
        st.markdown(new_title, unsafe_allow_html=True)
        a ,b= teachable_machine_classification(image, 'Tomato_disease_2.h5')
        if a == 0:
            new_title = '<p style="font-family:serif; color:black; font-size: 30px;border-radius:1000px ;background-color: #cbfc82; font-size: 30px;text-align: center"><b><u>Tomato Bacterial spot</u></b></p>'
            st.markdown(new_title, unsafe_allow_html=True)

        elif a == 1:
            new_title = '<p style="font-family:serif; color:black; font-size: 30px;border-radius:1000px ;background-color: #cbfc82; font-size: 30px;text-align: center"><b><u>Tomato Early Blight</u></b></p>'
            st.markdown(new_title, unsafe_allow_html=True)

        elif a == 2:
            new_title = '<p style="font-family:serif; color:black; font-size: 30px;border-radius:1000px ;background-color: #cbfc82; font-size: 30px;text-align: center;"><b><u><u>Tomato Late Blight</u></b></p>'
            st.markdown(new_title, unsafe_allow_html=True)

        elif a == 3:
            new_title = '<p style="font-family:serif; color:black; font-size: 30px;border-radius:1000px ;background-color:#cbfc82; font-size: 30px;text-align: center"><b><u>Tomato Leaf Mold</u></b></p>'
            st.markdown(new_title, unsafe_allow_html=True)

        elif a == 4:
            new_title = '<p style="font-family:serif; color:black; font-size: 30px;border-radius:1000px ;background-color: #cbfc82; font-size: 30px;text-align: center"><b><u>Tomato Septoria Leaf Spot</u></b></p>'
            st.markdown(new_title, unsafe_allow_html=True)

        elif a == 5:
            new_title = '<p style="font-family:serif; color:black; font-size: 30px; border-radius:1000px ;background-color: #cbfc82; font-size: 30px;text-align: center"><b><u>Tomato Spider Mites</u></b></p>'
            st.markdown(new_title, unsafe_allow_html=True)

        elif a == 6:
            new_title = '<p style="font-family:serif; color:black; font-size: 30px;border-radius:1000px ;background-color: #cbfc82; font-size: 30px;text-align:center"><b><u>Tomato Target Spot</u></b></p>'
            st.markdown(new_title, unsafe_allow_html=True)

        elif a == 7:
            new_title = '<p style="font-family:serif; color:black; font-size: 30px;border-radius:1000px ;background-color: #cbfc82; font-size: 30px;text-align: center"><b><u>Tomato Yellow Leaf Curl_Virus</u></b></p>'
            st.markdown(new_title, unsafe_allow_html=True)

        elif a == 8:
            new_title = '<p style="font-family:serif; color:black; font-size: 30px;border-radius:1000px ;background-color:#cbfc82; font-size: 30px;text-align: center"><b><u>Tomato Mosaic Virus</u></b></p>'
            st.markdown(new_title, unsafe_allow_html=True)

        elif a == 9:
            new_title = '<p style="font-family:serif; color:black; font-size: 30px; border-radius:1000px ;background-color: #cbfc82; font-size: 30px;text-align: center"><b><u>Tomato Healthy</u></b></p>'
            st.markdown(new_title, unsafe_allow_html=True)


        c=round(b*100,2)


        new_title = f'<p style="font-family:serif; color:black; border-radius:100px;background-color: #ade7f7; font-size: 30px;text-align: center"><b><u>Confidence->{c}</u></b></p>'
        st.markdown(new_title, unsafe_allow_html=True)







