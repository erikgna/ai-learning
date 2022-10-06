import os
from flask import Flask, render_template, request
from predicton import predict
from PIL import Image

app = Flask(__name__)

TYPE_VALUES = {'Carro': '2,40', 'Moto': '1,40', 'Caminhão': '3,40', 'Ônibus': '4,40'}

@app.route("/", methods=['POST', 'GET'])
def home():
    if request.method == 'POST':        
        content = request.files['image']

        img = Image.open(content)        
        converted = img.convert('RGB')
        resized = converted.resize([224,224])        

        prediction = predict(resized)

        path = None
        if prediction == 'Moto':
            path = 'static/moto/' + str(len(os.listdir('static/moto'))+1) + '.jpg'
            resized.save(path)
        if prediction == 'Carro':
            path = 'static/carro/' + str(len(os.listdir('static/carro'))+1) + '.jpg'
            resized.save(path)
        if prediction == 'Caminhão':
            path = 'static/caminhao/' + str(len(os.listdir('static/caminhao'))+1) + '.jpg'
            resized.save(path)
        if prediction == 'Ônibus':
            path = 'static/onibus/' + str(len(os.listdir('static/onibus'))+1) + '.jpg'
            resized.save(path) 

        return render_template('index.html', type=prediction, value=TYPE_VALUES[prediction], image=path)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)