import tkinter as tk
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageOps
from model import Model

model = Model()
model_path = Path(__file__).resolve().parent / "saved_models" / "mnist_model.pth"
model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
model.eval()
root = tk.Tk()
root.title("Digit Classifier")
root.geometry("380x480")
root.configure(bg="#202124")
root.resizable(False, False)
tk.Label(root, text="Digit Classifier", font=("Arial", 20, "bold"), fg="white", bg = "#202124")
canvas = tk.Canvas(
    root,
    width=280,
    height=280,
    bg="black",
    highlightthickness=0
)
canvas.pack()
image = Image.new("L", (280, 280), 0)
draw = ImageDraw.Draw(image)
last_position = None

def paint(event): 
    global last_position
    x, y = event.x, event.y
    if last_position is None:
        last_position = (x,y)
    x0, y0 = last_position
    canvas.create_line(
        x0, y0, x, y,
        fill="white",
        width=18,
        capstyle=tk.ROUND,
        smooth=True
    )
    draw.line((x0, y0, x, y), fill=255, width=18)
    draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=255)

    last_position = (x, y)

def start_drawing(event):
    global last_position
    last_position = (event.x, event.y)
    paint(event)

def stop_drawing(event):
    global last_position
    last_position = None


canvas.bind("<Button-1>", start_drawing)
canvas.bind("<B1-Motion>", paint)
canvas.bind("<ButtonRelease-1>", stop_drawing)


# Process the drawing and predict the digit
def predict_digit():
    # Find the bounding box around the handwriting
    bounds = image.getbbox()

    if bounds is None:
        result_label.config(text="Draw a digit first")
        return

    digit = image.crop(bounds)

    # Resize while preserving aspect ratio
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Put the digit in the center of a 28x28 image
    processed = Image.new("L", (28, 28), 0)
    x = (28 - digit.width) // 2
    y = (28 - digit.height) // 2
    processed.paste(digit, (x, y))

    # Convert to a tensor of shape (1, 1, 28, 28)
    import numpy as np

    pixels = np.array(processed, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, prediction].item() * 100

    result_label.config(
        text=f"Prediction: {prediction} ({confidence:.1f}%)"
    )


def clear_canvas():
    global image, draw, last_position

    canvas.delete("all")
    image = Image.new("L", (280, 280), 0)
    draw = ImageDraw.Draw(image)
    last_position = None

    result_label.config(text="Draw a digit")


# Create the prediction label
result_label = tk.Label(
    root,
    text="Draw a digit",
    font=("Arial", 18, "bold"),
    fg="white",
    bg="#202124"
)
result_label.pack(pady=15)


# Create the buttons
button_frame = tk.Frame(root, bg="#202124")
button_frame.pack()

clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_canvas,
    width=12
)
clear_button.pack(side=tk.LEFT, padx=8)

predict_button = tk.Button(
    button_frame,
    text="Predict",
    command=predict_digit,
    width=12,
    bg="white",
    fg="black"
)
predict_button.pack(side=tk.LEFT, padx=8)





    



root.mainloop()