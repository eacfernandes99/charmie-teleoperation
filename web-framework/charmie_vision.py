from flask import Flask,render_template,Response
import cv2
import numpy as np

app=Flask(__name__,template_folder='template')

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True: 
        photo = cap.read()[1]
        crop1 = photo[:480,0:550]
        crop2 = photo[:480,90:640]
        frame = np.concatenate([crop1, crop2], axis = 1)
        ret, buffer=cv2.imencode('.jpg', frame)
        frame=buffer.tobytes()

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

