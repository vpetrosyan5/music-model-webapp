import streamlit as st
from PIL import Image,ImageOps
from ultralytics import YOLO
import time

def load_model(model_name):
    model = YOLO(model_name)
    return model


# Deprecated - Not used
def music_generation_intf():
    file = st.file_uploader("Upload a WAV file as input",type=["wav","mp3"])
    if file:
        audio_bytes = file.read()
        st.audio(audio_bytes)


def music_detection_intf():
    st.sidebar.title("Music Detection")
    st.sidebar.header("Settings")

    models = {
        "Custom Music Detection Model":"models/model.pt",
        "Model 2":"models/davit_best.pt"
    }

    examples = {
        "Ունաբի":"examples/score_0.png",
        "Յոթ պար":"examples/score_1.jpg",
        "Մշո Շորոր":"examples/score_2.jpg",
        "Չինար ես":"examples/score_3.png"
    }

    all_classes =['black', 'key', 'line', 'piano', 'tone', 'voice', 'white']


    model_name = st.sidebar.selectbox("Select Model",models.keys())
    model = load_model(models[model_name])
    st.sidebar.subheader("Model Parameters")
    confidence = st.sidebar.slider("Confidence Threshold (%)", 0, 100, 30, 1)
    classes = st.sidebar.multiselect("Class Detection Filter",all_classes,default=all_classes)
    classids = [all_classes.index(cls) for cls in classes]
    show_labels = st.sidebar.checkbox("Show Labels")
    is_custom_img = st.sidebar.selectbox("Select Type",["Upload Image","Use Examples"]) == "Upload Image"

    if is_custom_img:
        uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = examples[st.sidebar.selectbox("Choose Example",examples.keys())]

    try:
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file).convert("RGB")

            #Remove EXIF data field that are in photos taken by phones that sometimes rotate the image
            uploaded_image = ImageOps.exif_transpose(uploaded_image)

            if st.sidebar.button('Start Detection'):
                with st.spinner("Please wait..."):
                    time.sleep(0.5)
                    res = model.predict(uploaded_image, conf=(confidence/100.0), imgsz=640,classes=classids)
                    col1, col2 = st.columns(2)
                    st.success("Done!")
                    with col1:
                        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
                    with col2:
                        st.image(res[0].plot(labels=show_labels), caption='Detected Image', use_container_width=True)
            else:
                    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        else:
            st.caption("Upload file or use examples to continue")
    except Exception as e:
        st.sidebar.error(f"An Error Occured: {e}")

def home_intf():
    col1,col2 = st.columns(2)

    with col1:
        st.image("resources/calfa-white.png",use_container_width=True)
    with col2:
        st.image("resources/tumo.png",use_container_width=True)
    st.write("This workshop is about reading and computing music using AI\nIn this webpage you can find some results of the workshop")


def music_reading_intf():
    musics = {
        "Չինար ես":{"audio":"reading/0.wav","image":"reading/0.jpg"},
        "Մշո Շորոր":{"audio":"reading/1.wav","image":"reading/1.jpg"}
    }
    sheet = st.sidebar.selectbox("Choose Image",musics.keys())
    
    image_path = musics[sheet]['image']
    wav_path = musics[sheet]['audio']
    
    st.caption("Selected Music Sheet - "+sheet)
    if st.sidebar.button("Start Generating",use_container_width=True):
        with st.spinner("Please Wait..."):
            time.sleep(5)
        st.success("Done!")
        st.audio(wav_path)
    st.image(image_path)    

def main_intf():
    st.sidebar.image("https://cardonstudios.com/wp-content/uploads/2015/08/1280px-Music_class_usa.jpg",use_container_width=True)
    st.sidebar.subheader("Workshop - Reading and computing music with AI")
    #st.sidebar.write("haha yes")

if __name__ == "__main__":
    actions = {
        "Home":home_intf,
        "Music Detection":music_detection_intf,
        "Music Reading":music_reading_intf
    }

    main_intf()

    action = st.selectbox("Action",actions.keys(),index = 0)
    actions[action]()