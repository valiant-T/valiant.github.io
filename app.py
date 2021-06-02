'''利用flask部署模型，并实现网页的预测返回结果'''
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import os
from PIL import Image
import matplotlib.pyplot as plt



# 获取项目当前绝对路径
# 比如我的项目在桌面上存放则得到——"C:\Users\asus\Desktop\shou"
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

# 加载模型
model = load_model('./models/plant_disease-mobileNet')

# 模型预测
def model_predict(img, model):
    # # img = img.resize((224, 224))
    # image_size = (224,224)
    # img = image.load_img(img,target_size=image_size)
    # # Preprocessing the image
    # x = image.img_to_array(img)
    # # x = np.true_divide(x, 255)
    # x = np.expand_dims(x, axis=0)

    preds = model.predict(img)
    return preds

def image_preprocess(img):
  new_shape = (224, 224)
  img = image.load_img(img,target_size=new_shape)
  image_array = image.img_to_array(img)
  # image_array = transform.resize(image_array, new_shape, anti_aliasing = True)
  image_array /= 255
  image_array = np.expand_dims(image_array, axis = 0)
  return image_array

@app.route('/',methods=['GET'])
def hello_world():
    # return 'Hello World!'
    return render_template('index.html')

@app.route('/getImg/',methods=['GET','POST'])
def getImg():
    # 通过表单中name值获取图片
    imgData = request.files["image"]
    # 设置图片要保存到的路径
    path = basedir + "/static/upload/img/"

    # 获取图片名称及后缀名
    imgName = imgData.filename

    # 图片path和名称组成图片的保存路径
    file_path = path + imgName

    # 保存图片
    imgData.save(file_path)

    # url是图片的路径
    # url = '/static/upload/img/' + imgName

    img_arr = image_preprocess(file_path)
    # 使用模型预测

    # result = model_predict(img_arr,model)
    predictions = model_predict(img_arr,model)
    # 显示类型
    li = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
          'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
          'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
          'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
          'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
          'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
          'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
          'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
          'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
          'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
          'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
          'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

    # print(li)

    d = predictions.flatten()
    j = d.max()
    for index, item in enumerate(d):
        if item == j:
            class_name = li[index]

    # print("result from model", result)
    print("result from model is :",class_name)
    # 等一下
    # return 'ok'
    # return result
    # return render_template('result.html',result_1 = result)
    return render_template('result.html',result_1 = class_name)





if __name__ == '__main__':
    # 测试 暂时不能用
    # img = 'E:/flaskproject/testimage/25.JPG'
    # result = model_predict(img,model)
    #
    # im = plt.imread(img)
    # plt.imshow(im)
    # plt.show()
    #
    # print(result)
    app.run(debug=True)
