import os
from flask import Flask, render_template, redirect, url_for, request, Response, flash, app
from werkzeug.utils import secure_filename
from show_prediction import ShowPrediction
# ------------------------------------------------------------ Configure Flask Application
app = Flask(__name__)
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

#------------------------------------------------------------Image Face Emotion Recognition
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in set([ 'png', 'jpg', 'jpeg', 'gif'])

@app.route('/upload_files')
def upload_form():
    return render_template('upload_images.html')

@app.route('/upload_files', methods=['POST'])
def upload_file():
    # Get current path
    path = os.getcwd()
    # file Upload
    UPLOAD_FOLDER = os.path.join('static', 'upload_image')

    # Make directory if uploads is not exists
    if not os.path.isdir(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    if request.method == 'POST':
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)

        files = request.files.getlist('files[]')
        file_names=[]
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_names.append(filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        flash('File(s) successfully uploaded')
        # image() - Return the images submitted by the user after identifying emotions, age, gender and race by face
        show_prediction.image()
        return render_template('upload_images.html', filenames=file_names)


#------------------------------------------------------------live_mp4 Face Emotion Recognition


def allowed_file_video( filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in set([ 'mp4', 'mp3'])

@app.route('/upload_files_videos')
def upload_form_video():
    return render_template('upload_videos.html')

@app.route('/upload_files_videos', methods=['POST'])
def upload_file_video():
    # Get current path
    path = os.getcwd()
    # video Upload
    UPLOAD_FOLDER_VIDEO = os.path.join('static', 'upload_video')

    # Make directory if uploads is not exists
    if not os.path.isdir(UPLOAD_FOLDER_VIDEO):
        os.mkdir(UPLOAD_FOLDER_VIDEO)

    app.config['UPLOAD_FOLDER_VIDEO'] = UPLOAD_FOLDER_VIDEO


    if request.method == 'POST':
        if 'file[]' not in request.files:
            flash('No file part')
            return redirect(request.url)

        files = request.files.getlist('file[]')
        file_names=[]
        for file in files:
            if file and allowed_file_video(file.filename):
                filename = secure_filename(file.filename)
                file_names.append(filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER_VIDEO'], filename))

        flash('File(s) successfully uploaded')
        # video() - Return the videos submitted by the user after identifying emotions, age, gender and race by face
        show_prediction.video()
        return render_template('upload_videos.html', filenames=file_names)

@app.route('/live_mp4_home')
def upload_live_mp4():
    return render_template('live_mp4_page.html')

@app.route('/live_mp4')
def get_live_mp4():
    return Response(show_prediction.video(),mimetype='multipart/x-mixed-replace; boundary=frame')

#------------------------------------------------------------live_camera Face Emotion Recognition

@app.route('/live_camera_home')
def upload_live_camera():
    return render_template('live_camera_page.html')

@app.route('/live_camera')
def get_live_camera():
    return Response(show_prediction.generate_frames_camera(),mimetype='multipart/x-mixed-replace; boundary=frame')

#------------------------------------------------------------HomePage Face Emotion Recognition
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/face',methods = ['POST', 'GET'])
def face():
    if request.method == 'POST':
        if request.form.get('video1') == 'video':
            return redirect(url_for('upload_live_mp4'))
        elif request.form.get('live_cam1') == 'live_cam':
            return redirect(url_for('upload_live_camera'))
        elif request.form.get('image1') == 'image':
            return redirect(url_for('upload_file'))
    return render_template("index.html")

    #------------------------------------------------------------


if __name__=='__main__':
    show_prediction = ShowPrediction()
    app.run(debug=True)

