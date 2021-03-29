from flask import Flask, render_template, request
import pickle
import pandas as pd

# initialize the flask app
app = Flask(__name__, template_folder="templates")
model = pickle.load(open("Diabetes.pkl", "rb"))


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


# Now predict function
@app.route('/predict', methods=['POST'])
def predict():
    # For rendering results on the HTML page
    prg = request.form['prg']
    glc = request.form['gl']
    bp = request.form['bp']
    skt = request.form['sk']
    ins = request.form['ins']
    bmi = request.form['BMI']
    dpf = request.form['ped']
    age = request.form['age']

    prg = int(prg)
    glc = int(glc)
    bp = int(bp)
    skt = int(skt)
    ins = int(ins)
    bmi = float(bmi)
    dpf = float(dpf)
    age = int(age)

    row_df = pd.DataFrame([pd.Series([prg, glc, bp, skt, ins, bmi,dpf, age])])
    #print(row_df)
    prediction = model.predict_proba(row_df)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    output = str(float(output) * 100) + '%'
    return render_template('index.html',pred=f'Probability of having Diabetes is {output}')



if __name__ == '__main__':
    app.run(debug=True)

