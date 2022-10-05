import os
from flask import Flask, render_template, request
from define import predict

app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        folder_info = os.listdir('uploads')        
        file_index = str(len(folder_info)+1)
        content = request.files['image']
        content.save('uploads/' + file_index + '.jpg')
        predict('uploads/' + file_index + '.jpg')
        return render_template('index.html')
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)