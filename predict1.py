import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import requests
import zipfile
import time
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 27)  
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


saved_model = CNN()
saved_model.load_state_dict(torch.load('fruit_classifier.pth'))
saved_model.eval()


new_image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


with open('class_labels.pth', 'rb') as f:
    class_labels = torch.load(f)


def predict_image_class(image, model, transform, class_labels):
    image = Image.open(image)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    class_index = predicted.item()
    class_name = class_labels[class_index]
    return class_name


def get_nutrition_details(fruit_name):
    url = f"https://www.fruityvice.com/api/fruit/{fruit_name.lower()}"
    response = requests.get(url)
    data = response.json()
    return data


def plot_pie_chart(nutrition_data):
    labels = ['Calories', 'Fat', 'Sugar', 'Carbohydrates', 'Protein']
    sizes = [nutrition_data['calories'], nutrition_data['fat'], nutrition_data['sugar'], 
             nutrition_data['carbohydrates'], nutrition_data['protein']]
    explode = (0.1, 0, 0, 0, 0)  
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    ax1.axis('equal')  
    st.pyplot(fig1)


def extract_dataset(uploaded_file):
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall('dataset')


def train_model(dataset_path):
    pass


st.set_page_config(page_title="EatLens", page_icon=":apple:")

st.title("EatLens")
st.image("Eatlens logo.png", width=300)
st.write("Scan food items to get their nutrition details.")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Test", "Model", "Details"])

if page == "Test":
    st.title("FruitLens")

    uploaded_file = st.file_uploader("Upload an image to Know the Details", type=["jpg", "png"])

    if uploaded_file is not None:
        st.write("Classifying...")
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            progress_bar.progress(percent_complete + 1)
            time.sleep(0.01)  
        predicted_class = predict_image_class(uploaded_file, saved_model, new_image_transform, class_labels)
        st.success("Predicted class: {}".format(predicted_class))
        
        nutrition_data = get_nutrition_details(predicted_class)
        if nutrition_data:
            st.subheader("Nutrition Details")
            st.write("Name:", nutrition_data['name'])
            st.write("Family:", nutrition_data['family'])
            st.write("Order:", nutrition_data['order'])
            st.write("Genus:", nutrition_data['genus'])
            st.subheader("Nutrition Information")
            st.write(f"<span style='color:white; font-size:20px; font-weight:bold;'>Calories:</span> <span style='color:white; font-size:20px; font-weight:bold;'>{nutrition_data['nutritions']['calories']}</span>", unsafe_allow_html=True)
            st.write(f"<span style='color:white; font-size:20px; font-weight:bold;'>Fat:</span> <span style='color:white; font-size:20px; font-weight:bold;'>{nutrition_data['nutritions']['fat']}</span>", unsafe_allow_html=True)
            st.write(f"<span style='color:white; font-size:20px; font-weight:bold;'>Sugar:</span> <span style='color:white; font-size:20px; font-weight:bold;'>{nutrition_data['nutritions']['sugar']}</span>", unsafe_allow_html=True)
            st.write(f"<span style='color:white; font-size:20px; font-weight:bold;'>Carbohydrates:</span> <span style='color:white; font-size:20px; font-weight:bold;'>{nutrition_data['nutritions']['carbohydrates']}</span>", unsafe_allow_html=True)
            st.write(f"<span style='color:white; font-size:20px; font-weight:bold;'>Protein:</span> <span style='color:white; font-size:20px; font-weight:bold;'>{nutrition_data['nutritions']['protein']}</span>", unsafe_allow_html=True)
            st.subheader("Nutrition Pie Chart")
            plot_pie_chart(nutrition_data['nutritions'])
            
        else:
            st.warning("Nutrition details not available for this fruit.")

elif page == "Model":
    st.subheader("Here is the flow of working of the project:")
    st.image("fruitlens.jpg", caption="Flow Diagram of Project")

   
    st.subheader("Training Loss and Accuracy")
    st.image("eatlens graph.png", caption="Training Loss and Validation Loss")
    st.write("Accuracy: 97%")
    

elif page == "Details":
    st.write("About the Project:")
    st.write("EatLens is a project aimed at providing nutritional information about various food items. It utilizes deep learning for image classification and an API for retrieving nutrition details.")
    st.write("Future Enhancements:")
    st.write("1. Integration of more sophisticated deep learning models.")
    st.write("2. Enhancing the UI/UX for better user experience.")
    st.image("Eatlensweb.png")
    st.write("3. Adding support for more food categories.")
    st.write("4. Optimizing the performance of the application.")
    st.write("GitHub Repository: [link](https://github.com/likhith1409/EatLens)")
    st.write("Dataset: [link](https://yourdatasetlink)")

