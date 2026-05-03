from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2


app = Flask(__name__)

dict = {0 : 'HC', 1 : 'PD'}

models = {
    'PET': load_model('PETmodelAlexNet-07-0.9985.hdf5'),
    'DAT': load_model('modeldatAlexnet-20-0.9958.hdf5'),
    'MRI': load_model('MRImodelAlexNet-94-0.9720.hdf5')
}

def predict_label(img_path, dataset):
	model = models[dataset]
	model.make_predict_function()
	#image = image.load_img(img_path, target_size=(227, 227))  # Resize image to match model input
	image = cv2.imread(img_path)

	if dataset == 'MRI':
		# Convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# Apply threshold to identify regions of high activity
		_, thresholded = cv2.threshold(gray, 196, 255, cv2.THRESH_BINARY)

		# Create a copy of the original image
		highlighted_image = gray.copy()

		# Apply red glow to regions of high activity
		highlighted_image[thresholded > 0] = 255  # Set high activity regions to maximum intensity

		# Convert the highlighted_image to a 3-channel format
		highlighted_image_rgb = cv2.cvtColor(highlighted_image, cv2.COLOR_GRAY2BGR)

		# Apply a different colormap to the highlighted regions for a different color effect
		custom_colormap = cv2.applyColorMap(highlighted_image_rgb, cv2.COLORMAP_HSV )  # Change colormap here

		# Combine the original grayscale image with the custom color effect
		image = cv2.addWeighted(image, 0.7, custom_colormap, 0.3, 0)


	image = cv2.resize(image, (227, 227))  # resize to the required dimensions
	#image = image.astype('float32') / 255  # normalize the image

	# Expand dimensions to match the input shape (1, height, width, channels)
	image = np.expand_dims(image, axis=0)

	# Predict the class
	y_pred_single = model.predict(image)
	predicted_class = np.argmax(y_pred_single, axis=1)

	print("predicted_label-->\n       HC          PD\n", y_pred_single)

	# Map the predicted class to the label
	return dict[predicted_class[0]], y_pred_single


@app.route("/", methods=['GET', 'POST'])
def home():
	return render_template('home.html')

@app.route("/submit", methods=['GET', 'POST'])
def predict():
	if request.method == 'POST':
		dataset = request.form['dataset']
		img = request.files['my_image']
		
		img_path = "static/" + img.filename
		img.save(img_path)

		p, y_pred_single = predict_label(img_path, dataset)

	return render_template('result.html', prediction = p, img_path = img_path, y_pred_single=y_pred_single)



if __name__ == '__main__':
	app.run(debug=True)