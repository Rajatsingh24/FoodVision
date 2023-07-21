import torch
import torchvision
import pickle
import gradio as gr
from timeit import default_timer as timer

# device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create the model 
model = pickle.load(open("model.pkl", "rb"))
model.to(device)
model.eval()
next(iter(model.parameters())).to(device)
# Transform for prediction
transform = torchvision.models.EfficientNet_B2_Weights.DEFAULT.transforms()
# class_names
# Open Food101 class names file and read each line into a list
with open("class_names.txt", "r") as f:
    class_names = [food.strip() for food in f.readlines()]

# example_list
example_list = [["example/pizza.jpg"],
                ["example/ice_cream.jpg"],
                ["example/pancakes.jpg"],
                ["example/steak.jpg"],
                ["example/hot_dog.jpg"]]


def predict(img):
    """Predict the class of image
    
    Args:
        img: input image vector
    Results:
        a tuple of dictionary and float
    Predictions,Prediction_time=predict(img=img_vector) 
    """
    start_time = timer()
    with torch.inference_mode():
        t_img = transform(img).unsqueeze(0).to(device)
        # print(t_img)
        probs = torch.softmax(model(t_img), dim=1).to("cpu")
    end_time = timer()
    label_and_prob = {class_names[i]: float(probs[0][i]) for i in range(len(class_names))}
    # print(probs)
    return label_and_prob, round(end_time - start_time, 4)


# Create title, description strings
title = "FoodVision 🍕🥩🍣"
description = "An EfficientNetB2 fine tuning computer vision model to classify 101 different food images."

# Create the Gradio demo
demo = gr.Interface(fn=predict,  # mapping function from input to output
                    inputs=gr.Image(type="pil"),  # what are the inputs?
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"),  # what are the outputs?
                             gr.Number(label="Prediction time(s)")],
                    # our fn has two outputs, therefore we have two outputs
                    examples=example_list,
                    title=title,
                    description=description)
# article=article)

demo.launch(debug=False)  # print errors locally?