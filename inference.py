import tkinter as tk
from tkinter import Canvas, Button, Label
from PIL import Image, ImageDraw
import numpy as np
import cv2
import openvino as ov
import os

def preprocess_drawing(image_gray):
    """
    Takes a large grayscale image of a black digit on a white background,
    finds the digit, centers it, and resizes it to the 28x28 MNIST format.
    """
    # Invert the image to get a white digit on a black background
    _, processed_image = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours to center the digit
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros((28, 28), dtype=np.float32)

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    digit_roi = processed_image[y:y+h, x:x+w]
    
    aspect_ratio = w / h
    if w > h:
        new_w, new_h = 20, int(20 / aspect_ratio)
    else:
        new_w, new_h = int(20 * aspect_ratio), 20
        
    resized_digit = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas_28x28 = np.zeros((28, 28), dtype=np.uint8)
    x_offset, y_offset = (28 - new_w) // 2, (28 - new_h) // 2
    canvas_28x28[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_digit
    
    return canvas_28x28.astype(np.float32) / 255.0

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        self.canvas = Canvas(root, width=280, height=280, bg="white", cursor="cross")
        self.canvas.pack(pady=10)

        self.predict_button = Button(root, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.clear_button = Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.RIGHT, padx=10, pady=10)

        self.prediction_label = Label(root, text="Draw a digit and click Predict", font=("Helvetica", 16))
        self.prediction_label.pack(pady=10)

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.last_x, self.last_y = None, None
        
        self.image = Image.new("RGB", (280, 280), "white")
        self.draw_tool = ImageDraw.Draw(self.image)

        self.load_model()

    def load_model(self):
        print("Loading model...")
        core = ov.Core()
        model_path = os.path.join("mnist_openvino_model", "best_model.xml")
        try:
            model = core.read_model(model=model_path)
            self.compiled_model = core.compile_model(model=model, device_name="CPU")
            self.input_layer = self.compiled_model.input(0)
            self.output_layer = self.compiled_model.output(0)
            print("Model loaded successfully! ✅")
        except Exception as e:
            self.prediction_label.config(text="Error: Model not found!")
            print(f"Error loading model: {e}")

    def start_drawing(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    width=20, fill="black", capstyle=tk.ROUND,
                                    smooth=tk.TRUE)
            self.draw_tool.line([self.last_x, self.last_y, event.x, event.y],
                                fill="black", width=20)
            self.last_x, self.last_y = event.x, event.y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_tool.rectangle([0, 0, 280, 280], fill="white")
        self.prediction_label.config(text="Draw a digit and click Predict")

    def predict_digit(self):
        img = self.image
        gray_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        processed_image_28x28 = preprocess_drawing(gray_image)
        
        # Display debug window
        debug_img_display = cv2.resize(processed_image_28x28, (280, 280), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("What the Model Sees", debug_img_display)

        # Reshape for the model
        input_tensor = np.expand_dims(np.expand_dims(processed_image_28x28, axis=0), axis=0)
        
        try:
            result = self.compiled_model([input_tensor])[self.output_layer]
            probabilities = np.exp(result)[0]
            predicted_digit = np.argmax(probabilities)
            confidence = np.max(probabilities)
            
            self.prediction_label.config(text=f"Prediction: {predicted_digit} (Conf: {confidence:.2%})")
            
            # --- THIS IS THE CHANGE: Print all probabilities to the console ---
            print("\n--- Prediction Breakdown ---")
            for i, prob in enumerate(probabilities):
                print(f"Digit {i}: {prob:.2%}")
            print(f"--------------------------\n-> Top Prediction: {predicted_digit} (Confidence: {confidence:.2%})\n")
            
        except Exception as e:
            error_message = "An error occurred during prediction."
            self.prediction_label.config(text=error_message)
            print(f"\n--- INFERENCE FAILED ---\n{e}\n------------------------\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()

