from flask import Flask, request, jsonify, render_template
import pickle



model = pickle.load(open("NavieBayesModel.pkl","rb"))
cv = pickle.load(open("vectorizer.pkl","rb"))
app = Flask(__name__,template_folder='template') # Creation of Flask Application

text =" "

@app.route("/") # This is Home Page  # Root Page /
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST']) # Providing Features as input to model and providing output
def predict(): # Web API
    '''
    For Rendering Results from on HTML GUI
    '''
    if request.method == "POST":
        msg = request.form["message"]
        data = [msg]
        vect = cv.transform(data).toarray()
        pred = model.predict(vect)
    pred = pred[0]
    if(pred==0):
        return render_template('index.html',prediction_text="No Clickbait")
    else:
        return render_template('index.html',prediction_text="Clickbait")

if __name__ == "__main__":
    app.run(debug=True)