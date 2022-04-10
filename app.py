
from re import T
from flask import Flask, render_template, request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from flask_sqlalchemy import SQLAlchemy
import sqlite3
import base64
import io
from PIL import Image

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI']= "sqlite:///anaemic.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']= False
db = SQLAlchemy(app)

class Upload(db.Model):
	id= db.Column(db.Integer, primary_key=True)
	filename= db.Column(db.String(50))
	data= db.Column(db.LargeBinary)

def get_db_connection():
    conn = sqlite3.connect('anaemic.db')
    conn.row_factory = sqlite3.Row
    return conn

def writeTofile(data,filename):
	with open(filename,'wb') as file:
		a=file.write(data)
		return a
	print('Sorted blob data into: ', filename, '\n')

@app.route("/")
@app.route("/home")
def home():
    return render_template("image_upload.html")


@app.route("/result", methods=['POST'])
def result():
    print("abcd")
    f = request.files['file']
    file = base64.b64encode(f.read())
    upload= Upload(filename=f.filename, data= file)
    db.session.add(upload)
    db.session.commit()
    print(f'Uploaded:{f.filename}')
    #f.save("C:\\Users\\BHAVIT JAIN\\Desktop\\FlaskApp"+"\\test.jpg")
    
    return model()

def model():
	print("1")
	connect = get_db_connection()
	record = connect.execute('SELECT * FROM upload ORDER BY id DESC LIMIT 1').fetchall()
	for row in record:
		print('ID:',row[0], 'filename:', row[1])
		name=row[1]
		e_img= row[2]
		binary_data = base64.b64decode(e_img)
		image = Image.open(io.BytesIO(binary_data))
		image.show()
		#print(eyes.format)
		#e_img_path= 'C:\\Users\\BHAVIT JAIN\\Desktop\\FlaskApp\\testimg.jpg'
		#writeTofile(e_img,e_img_path)

	#connect.close()
	#print('conn closed')


	#__,enc = cv2.imencode('.jpg',e_img)
	#connect.execute("insert into images values(?,?)",("patient",buffer(enc)))
	#db.commit()
	#image = cv2.imread('C:\\Users\\BHAVIT JAIN\\Desktop\\FlaskApp\\testimg.jpg')
	#image.show()
	# create a mask image of the same shape as input image, filled with 0s (black color)
	image= np.array(image)
	mask = np.zeros_like(image)
	rows, cols,_ = mask.shape
	# create a white filled ellipse
	print(2)
	mask=cv2.ellipse(mask, center=(110, 176), axes=(35, 15), angle=0, startAngle=0, endAngle=360, color=(255,255,255), thickness=-1)
	# Bitwise AND operation to black out regions outside the mask
	result = np.bitwise_and(image,mask)
	# Convert from BGR to RGB for displaying correctly in matplotlib
	# Note that you needn't do this for displaying using OpenCV's imshow()
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	print(3)
	mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
	result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
	# Plotting the results
	#plt.subplot(131)
	#plt.imshow(image_rgb);
	#plt.subplot(132)
	#plt.imshow(mask_rgb)
	#plt.subplot(133)
	#plt.imshow(result_rgb);
	#plt.show()
	rgb_img = cv2.cvtColor(result_rgb, cv2.COLOR_BGR2RGB) #cropped image is stored in result_rgb
	#cv2.imshow("Img",rgb_img)
	#cv2.waitKey(0)
	print(4)
	red_pixels = rgb_img[:,:,2]
	green_pixels = rgb_img[:,:,1]
	blue_pixels = rgb_img[:,:,0]

	#Performing operations on red pixels of the extracted cropped image.
	red_pixels_ar = np.array(red_pixels)
	red_pixels_1 = np.size(np.array(red_pixels))
	red_pixels_count = np.sum(np.array(red_pixels)>0)

	print(5)
	sum_red_pixels = np.sum(red_pixels_ar) 
	print(6)
	mean_red_pixel_intensity= sum_red_pixels/red_pixels_count
	#Performing operations on green pixels of the extracted cropped image.
	green_pixels_ar = np.array(green_pixels)
	green_pixels_1 = np.size(np.array(green_pixels))
	green_pixels_count = np.sum(np.array(green_pixels)>0)
	print(7)
	sum_green_pixels = np.sum(green_pixels_ar) 
	print(8)
	mean_green_pixel_intensity= sum_green_pixels/green_pixels_count

	diff_pixels = mean_red_pixel_intensity-mean_green_pixel_intensity

	


	print('Status of Patient :')
	if (diff_pixels > 59):
	    return "Non-Anaemic"
	else:
	    return "Anaemic" 

if __name__ == '__main__':
    app.run(debug=True, port=5001)

